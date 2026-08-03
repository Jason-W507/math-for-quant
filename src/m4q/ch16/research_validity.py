from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys


UNIVERSAL_FRICTION_FIELDS = {
    "point_in_time_calendar",
    "commission",
    "spread",
    "market_impact",
    "borrow_and_short_availability",
}
A_SHARE_CONDITIONAL_FIELDS = {
    "t_plus_one_inventory",
    "price_limit_fill_model",
    "suspension_calendar",
    "point_in_time_corporate_actions",
    "short_sale_eligibility",
}


from m4q.evidence import load_oracle_bundle

def validate_protocol(
    label: str, protocol: dict[str, bool], required_fields: set[str]
) -> None:
    missing = sorted(required_fields - set(protocol))
    if missing:
        raise SystemExit(f"friction protocol failed: missing {label} field {missing[0]}")
    unexpected = sorted(set(protocol) - required_fields)
    if unexpected:
        raise SystemExit(
            f"friction protocol failed: unexpected {label} field {unexpected[0]}"
        )
    disabled = sorted(field for field in required_fields if protocol[field] is not True)
    if disabled:
        raise SystemExit(f"friction protocol failed: {label} field {disabled[0]} disabled")


def main(oracle_path: Path) -> int:
    oracle = load_oracle_bundle(oracle_path)
    tolerance = float(oracle["absolute_tolerance"])

    training_end = date.fromisoformat(oracle["training_end"])
    test_start = date.fromisoformat(oracle["test_start"])
    if training_end >= test_start:
        raise SystemExit("split gate failed: training period overlaps the test period")

    timezone_name = oracle.get("market_timezone")
    offset_text = oracle.get("market_utc_offset")
    if timezone_name != "Asia/Shanghai" or offset_text != "+08:00":
        raise SystemExit("timezone gate failed: expected Asia/Shanghai (+08:00)")
    expected_offset = timedelta(hours=8)

    calendar = oracle.get("trading_calendar", {})
    calendar_version = calendar.get("version")
    trading_dates = set(calendar.get("trading_dates", []))
    if not calendar.get("id") or not calendar_version or not trading_dates:
        raise SystemExit("calendar gate failed: missing versioned trading calendar")

    rows = oracle["test_rows"]
    for index, row in enumerate(rows, start=1):
        parsed = {}
        for field in ("event_at", "available_at", "decision_at", "target_start"):
            parsed[field] = datetime.fromisoformat(row[field])
            if parsed[field].utcoffset() != expected_offset:
                raise SystemExit(
                    f"timezone gate failed: row {index} {field} is not "
                    "Asia/Shanghai (+08:00)"
                )
        event_at = parsed["event_at"]
        available_at = parsed["available_at"]
        decision_at = parsed["decision_at"]
        target_start = parsed["target_start"]
        if not event_at <= available_at <= decision_at:
            raise SystemExit(
                f"timeline gate failed: row {index} was not available before the decision"
            )
        if not decision_at < target_start:
            raise SystemExit(
                f"timeline gate failed: row {index} target begins before the decision"
            )
        if decision_at.date() < test_start:
            raise SystemExit(f"split gate failed: row {index} is not in the test period")
        if decision_at.date().isoformat() not in trading_dates:
            raise SystemExit(
                f"calendar gate failed: row {index} decision date is not in "
                f"calendar {calendar_version}"
            )

    tested_hypotheses = int(oracle["tested_hypotheses"])
    threshold = float(oracle["family_alpha"]) / tested_hypotheses
    selected_p = float(oracle["selected_raw_p_value"])
    if selected_p > threshold:
        raise SystemExit(
            f"multiple-testing gate failed: selected p-value exceeds {threshold:.6f}"
        )

    costs = oracle["cost_components_per_trade"]
    cost_per_trade = sum(float(value) for value in costs.values())
    gross_return = sum(float(row["gross_return"]) for row in rows)
    total_cost = cost_per_trade * len(rows)
    net_return = gross_return - total_cost
    expected = oracle["expected"]

    universal_fields = oracle["universal_friction_protocol"]
    a_share_fields = oracle["a_share_conditional_protocol"]
    validate_protocol("universal", universal_fields, UNIVERSAL_FRICTION_FIELDS)
    validate_protocol(
        "A-share conditional", a_share_fields, A_SHARE_CONDITIONAL_FIELDS
    )
    checks = [
        tested_hypotheses > 1,
        abs(threshold - float(expected["bonferroni_threshold"])) <= tolerance,
        abs(gross_return - float(expected["gross_return"])) <= tolerance,
        abs(total_cost - float(expected["total_cost"])) <= tolerance,
        abs(net_return - float(expected["net_return"])) <= tolerance,
        set(costs) == {"commission", "spread", "impact"},
    ]
    if not all(checks):
        raise SystemExit("research-validity oracle or friction protocol failed")

    print(
        "oracle=passed timeline=passed timezone=Asia/Shanghai "
        f"calendar={calendar_version} split=passed "
        f"multiplicity=(tests={tested_hypotheses},threshold={threshold:.6f},"
        f"p={selected_p:.6f}) "
        f"performance=(gross={gross_return:.6f},cost={total_cost:.6f},"
        f"net={net_return:.6f}) "
        "frictions=passed a_share=conditional"
    )
    return 0


if __name__ == "__main__":
    oracle_path = Path("evidence/ch16/oracle.json")
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        oracle_path = Path(sys.argv[1])
    main(oracle_path)
