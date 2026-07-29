from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from math_for_quant.evidence import load_oracle_bundle


def validate_covariance(covariance: np.ndarray) -> None:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-12:
        raise ValueError("covariance must be positive semidefinite")


def factor_covariance(loadings: np.ndarray, factor_variance: np.ndarray, idiosyncratic_variance: np.ndarray) -> np.ndarray:
    if loadings.shape[1] != factor_variance.shape[0] or factor_variance.shape[0] != factor_variance.shape[1]:
        raise ValueError("factor covariance dimensions disagree")
    if loadings.shape[0] != idiosyncratic_variance.size:
        raise ValueError("idiosyncratic variance dimensions disagree")
    covariance = loadings @ factor_variance @ loadings.T + np.diag(idiosyncratic_variance)
    validate_covariance(covariance)
    return covariance


def minimum_variance_two_asset(covariance: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    validate_covariance(covariance)
    if covariance.shape != (2, 2):
        raise ValueError("hand-solution oracle requires two assets")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    regularized = covariance + ridge * np.eye(2)
    denominator = regularized[0, 0] + regularized[1, 1] - 2.0 * regularized[0, 1]
    if denominator <= 0.0:
        raise ValueError("minimum-variance denominator must be positive")
    first = float(np.clip((regularized[1, 1] - regularized[0, 1]) / denominator, 0.0, 1.0))
    return np.array([first, 1.0 - first])


def cost_aware_rebalance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    current_weights: np.ndarray,
    *,
    risk_aversion: float,
    cost_rate: float,
    capital: float,
    maximum_weight: float,
    tradable: np.ndarray,
    grid_step: float = 0.1,
) -> tuple[np.ndarray, float, float]:
    validate_covariance(covariance)
    if any(array.shape != (2,) for array in (expected_returns, current_weights, tradable)):
        raise ValueError("rebalance oracle requires two assets")
    if np.any((tradable == 0) & (np.abs(current_weights - np.round(current_weights / grid_step) * grid_step) > 1e-12)):
        raise ValueError("current weights must lie on the search grid")
    best_score = -float("inf")
    best = current_weights.copy()
    points = int(round(1.0 / grid_step))
    for index in range(points + 1):
        candidate = np.array([index * grid_step, 1.0 - index * grid_step])
        if np.any(candidate < -1e-12) or np.any(candidate > maximum_weight + 1e-12):
            continue
        if np.any((tradable == 0) & (np.abs(candidate - current_weights) > 1e-12)):
            continue
        turnover = float(np.abs(candidate - current_weights).sum())
        score = float(candidate @ expected_returns - 0.5 * risk_aversion * candidate @ covariance @ candidate - cost_rate * turnover)
        if score > best_score + 1e-15:
            best_score, best = score, candidate
    if not np.isfinite(best_score):
        raise ValueError("no feasible portfolio under tradability and weight constraints")
    turnover = float(np.abs(best - current_weights).sum())
    return best, turnover, capital * cost_rate * turnover


def reject_untradable_change(current: np.ndarray, proposed: np.ndarray, tradable: np.ndarray) -> None:
    if np.any((tradable == 0) & (np.abs(proposed - current) > 1e-12)):
        raise ValueError("untradable asset weight cannot change")


def empirical_var_es(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    if losses.ndim != 1 or losses.size < 4:
        raise ValueError("tail sample requires at least four losses")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    ordered = np.sort(losses)
    index = int(np.ceil(confidence * ordered.size)) - 1
    value_at_risk = float(ordered[index])
    tail = ordered[index + 1 :]
    if tail.size == 0:
        raise ValueError("tail sample is empty at this confidence")
    return value_at_risk, float(tail.mean())


def expect_rejection(action, diagnostic: str) -> int:
    try:
        action()
    except ValueError as error:
        if diagnostic not in str(error):
            raise AssertionError(f"wrong diagnostic: {error}") from error
        return 1
    raise AssertionError(f"expected rejection containing {diagnostic!r}")


def render_report(observed: dict[str, float | int]) -> str:
    return f"""# 组合与风险可复现研究包

- 协方差 oracle：$\\sigma_1^2={observed['variance_1']:.6f}$，$\\sigma_{{12}}={observed['covariance_12']:.6f}$，$\\sigma_2^2={observed['variance_2']:.6f}$；因子分解与直接矩阵一致。
- 优化：二资产最小方差权重为 ({observed['minimum_weight_1']:.6f}, {observed['minimum_weight_2']:.6f})；加入协方差 ridge 后为 ({observed['robust_weight_1']:.6f}, {observed['robust_weight_2']:.6f})，说明输入扰动会改变解。
- 再平衡台账：目标权重 ({observed['rebalance_weight_1']:.6f}, {observed['rebalance_weight_2']:.6f})，双边换手 {observed['turnover']:.6f}，线性成本 {observed['cost_cash']:.2f}；最大权重与可交易约束在搜索中生效。
- 风险：75% 经验 VaR 为 {observed['var']:.6f}，严格尾部 ES 为 {observed['es']:.6f}，压力情景损失为 {observed['stress_loss']:.6f}。
- 失败边界：非正定协方差、不可交易资产变仓、尾部样本不足分别拒绝；正态轻尾近似不能替代经验尾部和压力情景。
- 限制：两资产网格不是生产优化器；线性成本忽略冲击与容量；历史 VaR/ES 不保证未来尾部稳定。
- 复现命令：`uv run python notebooks/lower/ch05_portfolio_risk.py evidence/lower-ch05/oracle.json`。
"""


def main(oracle_path: Path = Path("evidence/lower-ch05/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    direct = np.asarray(oracle["covariance"], dtype=float)
    modeled = factor_covariance(
        np.asarray(oracle["factor_loadings"], dtype=float),
        np.asarray(oracle["factor_variance"], dtype=float),
        np.asarray(oracle["idiosyncratic_variance"], dtype=float),
    )
    if not np.allclose(direct, modeled, atol=float(oracle["absolute_tolerance"])):
        raise SystemExit("factor covariance disagrees with direct oracle")
    minimum = minimum_variance_two_asset(direct)
    robust = minimum_variance_two_asset(direct, float(oracle["ridge"]))
    weights, turnover, cost_cash = cost_aware_rebalance(
        np.asarray(oracle["expected_returns"], dtype=float),
        direct,
        np.asarray(oracle["current_weights"], dtype=float),
        risk_aversion=float(oracle["risk_aversion"]),
        cost_rate=float(oracle["cost_rate"]),
        capital=float(oracle["capital"]),
        maximum_weight=float(oracle["maximum_weight"]),
        tradable=np.asarray(oracle["tradable"], dtype=int),
        grid_step=float(oracle["grid_step"]),
    )
    var, es = empirical_var_es(np.asarray(oracle["losses"], dtype=float), float(oracle["confidence"]))
    stress_loss = float(np.asarray(oracle["stress_shocks"], dtype=float) @ weights * -1.0)
    failures = (
        expect_rejection(lambda: validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]])), "positive semidefinite"),
        expect_rejection(lambda: reject_untradable_change(np.array([0.4, 0.6]), np.array([0.5, 0.5]), np.array([0, 1])), "untradable"),
        expect_rejection(lambda: empirical_var_es(np.array([1.0, 2.0, 3.0]), 0.95), "at least four"),
    )
    observed: dict[str, float | int] = {
        "variance_1": direct[0, 0], "covariance_12": direct[0, 1], "variance_2": direct[1, 1],
        "minimum_weight_1": minimum[0], "minimum_weight_2": minimum[1],
        "robust_weight_1": robust[0], "robust_weight_2": robust[1],
        "rebalance_weight_1": weights[0], "rebalance_weight_2": weights[1],
        "turnover": turnover, "cost_cash": cost_cash, "var": var, "es": es, "stress_loss": stress_loss,
        "covariance_rejected": failures[0], "untradable_rejected": failures[1], "tail_sample_rejected": failures[2],
    }
    tolerance = float(oracle["absolute_tolerance"])
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected:
                raise SystemExit(f"{name} failed: {value} != {expected}")
        elif abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} failed: {value} != {expected}")
    report_path = Path(oracle["report"])
    if report_path.read_text(encoding="utf-8") != render_report(observed):
        raise SystemExit(f"reproducible report drifted: {report_path}")
    print(
        "oracle=passed "
        f"covariance=({direct[0,0]:.6f},{direct[0,1]:.6f},{direct[1,1]:.6f}) "
        f"weights=({minimum[0]:.6f},{minimum[1]:.6f},{robust[0]:.6f},{robust[1]:.6f}) "
        f"rebalance=({weights[0]:.6f},{weights[1]:.6f},{turnover:.6f},{cost_cash:.6f}) "
        f"risk=({var:.6f},{es:.6f},{stress_loss:.6f}) failures=({','.join(str(value) for value in failures)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch05/oracle.json")))
