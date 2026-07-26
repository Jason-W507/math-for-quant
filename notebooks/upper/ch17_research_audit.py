# %% [markdown]
# # Capstone：可复现研究审计
#
# 独立 oracle 固定数据校验值、时间边界、多重检验、成本账本、数值探针、许可与限制；审计程序只负责拒绝不满足声明的研究包。

# %%
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys


REQUIRED_COLUMNS = {
    "observation_id",
    "event_at",
    "available_at",
    "decision_at",
    "target_start",
    "gross_return",
}
REQUIRED_COSTS = {"commission", "spread", "impact"}
REQUIRED_LICENSES = {
    "data": "CC0-1.0",
    "code": "MIT",
    "manuscript": "CC-BY-NC-SA-4.0",
    "template": "LPPL-1.3c",
}


def fail(message: str) -> None:
    raise SystemExit(message)


def audit(oracle_path: Path, package_root: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    tolerance = float(oracle["absolute_tolerance"])

    environment = oracle["environment"]
    if tuple(sys.version_info[:2]) < tuple(environment["python_minimum"]):
        fail("environment gate failed: Python version is below declared minimum")
    if environment["research_runtime"] != "Python standard library":
        fail("environment gate failed: research runtime is not reproducible")
    for relative in oracle["package_files"]:
        if not (package_root / relative).is_file():
            fail(f"package gate failed: missing declared file {relative}")

    data_path = package_root / oracle["data_path"]
    if not data_path.is_file():
        fail("data gate failed: declared research input is missing")
    observed_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if observed_hash != oracle["data_sha256"]:
        fail("data gate failed: research input checksum mismatch")

    with data_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != REQUIRED_COLUMNS:
            fail("data gate failed: research input schema mismatch")
        rows = list(reader)
    expected = oracle["expected"]
    if len(rows) != int(expected["rows"]):
        fail("data gate failed: research row count mismatch")

    timezone_name = oracle["market_timezone"]
    offset_text = oracle["market_utc_offset"]
    if timezone_name != "Asia/Shanghai" or offset_text != "+08:00":
        fail("timezone gate failed: expected Asia/Shanghai (+08:00)")
    expected_offset = timedelta(hours=8)

    calendar = oracle["trading_calendar"]
    calendar_version = calendar["version"]
    trading_dates = set(calendar["trading_dates"])
    training_end = date.fromisoformat(oracle["training_end"])
    test_start = date.fromisoformat(oracle["test_start"])
    if training_end >= test_start:
        fail("split gate failed: training period overlaps the final test")

    gross_contributions: list[float] = []
    for index, row in enumerate(rows, start=1):
        times: dict[str, datetime] = {}
        for field in ("event_at", "available_at", "decision_at", "target_start"):
            times[field] = datetime.fromisoformat(row[field])
            if times[field].utcoffset() != expected_offset:
                fail(f"timezone gate failed: row {index} {field} has wrong offset")
        if not times["event_at"] <= times["available_at"] <= times["decision_at"]:
            fail(f"timeline gate failed: row {index} was unavailable at decision")
        if not times["decision_at"] < times["target_start"]:
            fail(f"timeline gate failed: row {index} target begins too early")
        decision_date = times["decision_at"].date()
        if decision_date < test_start:
            fail(f"split gate failed: row {index} is not in the final test")
        if decision_date.isoformat() not in trading_dates:
            fail(
                f"calendar gate failed: row {index} is outside {calendar_version}"
            )
        gross_contributions.append(float(row["gross_return"]))

    tested_hypotheses = int(oracle["tested_hypotheses"])
    threshold = float(oracle["family_alpha"]) / tested_hypotheses
    selected_p = float(oracle["selected_raw_p_value"])
    if selected_p > threshold:
        fail("multiplicity gate failed: selected result exceeds corrected threshold")

    costs = oracle["cost_components_per_trade"]
    if set(costs) != REQUIRED_COSTS:
        fail("friction gate failed: cost component schema mismatch")
    gross_return = math.fsum(gross_contributions)
    total_cost = len(rows) * math.fsum(float(value) for value in costs.values())
    net_return = gross_return - total_cost
    ledger_checks = (
        abs(threshold - float(expected["bonferroni_threshold"])) <= tolerance,
        abs(gross_return - float(expected["gross_return"])) <= tolerance,
        abs(total_cost - float(expected["total_cost"])) <= tolerance,
        abs(net_return - float(expected["net_return"])) <= tolerance,
    )
    if not all(ledger_checks):
        fail("performance gate failed: independent ledger mismatch")

    probe = [float(value) for value in oracle["numeric_probe"]]
    naive_probe_sum = 0.0
    for value in probe:
        naive_probe_sum += value
    if naive_probe_sum != float(expected["naive_probe_sum"]):
        fail("numeric gate failed: cancellation probe no longer exposes naive sum")
    if math.fsum(probe) != float(expected["stable_probe_sum"]):
        fail("numeric gate failed: stable sum does not match independent oracle")

    license_manifest_path = package_root / oracle["license_manifest_path"]
    if not license_manifest_path.is_file():
        fail("license gate failed: asset license manifest is missing")
    license_manifest_bytes = license_manifest_path.read_bytes()
    if (
        hashlib.sha256(license_manifest_bytes).hexdigest()
        != oracle["license_manifest_sha256"]
    ):
        fail("license gate failed: asset license manifest checksum mismatch")
    license_manifest = json.loads(license_manifest_bytes)
    licensed_assets = license_manifest.get("assets", [])
    if not isinstance(licensed_assets, list):
        fail("license gate failed: asset records must be a list")
    if not all(isinstance(asset, dict) for asset in licensed_assets):
        fail("license gate failed: asset record is invalid")
    asset_ids = [asset.get("id") for asset in licensed_assets]
    if not all(isinstance(asset_id, str) and asset_id for asset_id in asset_ids):
        fail("license gate failed: asset class is invalid")
    if len(asset_ids) != len(set(asset_ids)):
        fail("license gate failed: duplicate asset class")
    if set(asset_ids) != set(REQUIRED_LICENSES):
        fail("license gate failed: required asset classes are incomplete")
    for asset in licensed_assets:
        asset_id = str(asset["id"])
        if asset.get("license_id") != REQUIRED_LICENSES[asset_id]:
            fail(f"license gate failed: {asset_id} license id is invalid")
        scope = str(asset.get("scope", "")).strip()
        if not scope:
            fail(f"license gate failed: {asset_id} scope is missing")
        paths = asset.get("paths")
        if not isinstance(paths, list) or not paths or not all(
            isinstance(relative, str) and relative.strip() for relative in paths
        ):
            fail(f"license gate failed: {asset_id} asset paths are missing")
        license_file = asset.get("license_file")
        if not isinstance(license_file, str) or not license_file.strip():
            fail(f"license gate failed: {asset_id} license file is missing")
        required_marker = asset.get("required_marker")
        if not isinstance(required_marker, str) or not required_marker.strip():
            fail(f"license gate failed: {asset_id} license marker is missing")
        for relative in paths:
            if not (package_root / relative).is_file():
                fail(f"license gate failed: {asset_id} asset is missing")
        license_path = package_root / license_file
        if not license_path.is_file() or required_marker not in (
            license_path.read_text(encoding="utf-8")
        ):
            fail(f"license gate failed: {asset_id} license marker is missing")

    limitations_path = package_root / oracle["limitations_path"]
    if not limitations_path.is_file():
        fail("report gate failed: limitation report is missing")
    limitations_bytes = limitations_path.read_bytes()
    if hashlib.sha256(limitations_bytes).hexdigest() != oracle["limitations_sha256"]:
        fail("report gate failed: limitation report checksum mismatch")
    limitations = [
        line
        for line in limitations_bytes.decode("utf-8").splitlines()
        if line.startswith("- ")
    ]
    if len(limitations) != int(oracle["expected_limitations"]):
        fail("report gate failed: limitation count mismatch")

    print(
        f"oracle=passed audit=passed package={oracle['package_id']} "
        f"data={data_path.name} rows={len(rows)} timeline=passed split=passed "
        "multiplicity=passed "
        f"performance=(gross={gross_return:.6f},cost={total_cost:.6f},"
        f"net={net_return:.6f}) numeric=passed "
        f"license={oracle['license_id']} licenses={len(licensed_assets)} "
        f"limitations={len(limitations)}"
    )
    return 0


root = Path.cwd()
oracle = root / "evidence" / "ch17" / "oracle.json"
if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    oracle = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        root = Path(sys.argv[2]).resolve()
audit(oracle, root)
