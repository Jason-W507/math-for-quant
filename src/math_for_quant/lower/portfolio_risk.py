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
    validate_covariance(factor_variance)
    if np.any(idiosyncratic_variance < 0.0):
        raise ValueError("idiosyncratic variances must be nonnegative")
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
    if risk_aversion <= 0.0:
        raise ValueError("risk aversion must be positive")
    if capital <= 0.0:
        raise ValueError("capital must be positive")
    if cost_rate < 0.0:
        raise ValueError("cost rate must be nonnegative")
    if grid_step <= 0.0 or not np.isclose(round(1.0 / grid_step) * grid_step, 1.0):
        raise ValueError("grid step must divide one exactly")
    if not np.isclose(current_weights.sum(), 1.0, atol=1e-12):
        raise ValueError("current weights must sum to one")
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


def empirical_var_es(
    losses: np.ndarray,
    confidence: float,
    *,
    minimum_tail_observations: int = 20,
) -> tuple[float, float]:
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("tail sample requires a nonempty one-dimensional loss array")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if minimum_tail_observations <= 0:
        raise ValueError("minimum effective tail observations must be positive")
    effective_tail = losses.size * (1.0 - confidence)
    if effective_tail + 1e-12 < minimum_tail_observations:
        raise ValueError(
            "insufficient effective tail observations: "
            f"observed={effective_tail:.6g}, required={minimum_tail_observations}"
        )
    ordered = np.sort(losses)
    index = int(np.ceil(confidence * ordered.size)) - 1
    value_at_risk = float(ordered[index])
    tail_mass = (1.0 - confidence) * ordered.size
    full_count = int(np.floor(tail_mass))
    fractional = tail_mass - full_count
    tail_sum = float(ordered[-full_count:].sum()) if full_count else 0.0
    if fractional > 1e-15:
        tail_sum += fractional * float(ordered[-full_count - 1])
    if tail_mass <= 0.0:
        raise ValueError("tail sample is empty at this confidence")
    return value_at_risk, tail_sum / tail_mass


def expect_rejection(action, diagnostic: str) -> int:
    try:
        action()
    except ValueError as error:
        if diagnostic not in str(error):
            raise AssertionError(f"wrong diagnostic: {error}") from error
        return 1
    raise AssertionError(f"expected rejection containing {diagnostic!r}")


def render_experiment_contract(observed: dict[str, float | int]) -> str:
    return (
        f"confidence={observed['confidence']:.4f}，ridge={observed['ridge']:.6f}，"
        f"risk_aversion={observed['risk_aversion']:.4f}，cost_rate={observed['cost_rate']:.6f}，"
        f"maximum_weight={observed['maximum_weight']:.4f}，grid_step={observed['grid_step']:.4f}，"
        f"tradable_assets={int(observed['tradable_assets'])}"
    )


def render_report(observed: dict[str, float | int]) -> str:
    contract = render_experiment_contract(observed)
    return f"""# 组合与风险可复现研究包

- 协方差 oracle：$\\sigma_1^2={observed['variance_1']:.6f}$，$\\sigma_{{12}}={observed['covariance_12']:.6f}$，$\\sigma_2^2={observed['variance_2']:.6f}$；因子分解与直接矩阵一致。
- 优化：二资产最小方差权重为 ({observed['minimum_weight_1']:.6f}, {observed['minimum_weight_2']:.6f})。病态基准、小扰动和 ridge 稳健处理的第一资产权重依次为 {observed['ill_weight']:.6f}、{observed['perturbed_weight']:.6f}、{observed['stabilized_weight']:.6f}。
- 再平衡台账：目标权重 ({observed['rebalance_weight_1']:.6f}, {observed['rebalance_weight_2']:.6f})，双边换手 {observed['turnover']:.6f}，线性成本 {observed['cost_cash']:.2f}；最大权重在基准搜索中活跃，不可交易冻结由独立无可行解负例验证。
- 风险：置信水平 {observed['confidence']:.2%} 的经验 VaR 为 {observed['var']:.6f}，积分分位数 ES 为 {observed['es']:.6f}，压力情景损失为 {observed['stress_loss']:.6f}。
- 实验契约：{contract}。
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
    ill_conditioned = np.asarray(oracle["ill_conditioned_covariance"], dtype=float)
    perturbed = np.asarray(oracle["perturbed_covariance"], dtype=float)
    ill_weight = minimum_variance_two_asset(ill_conditioned)[0]
    perturbed_weight = minimum_variance_two_asset(perturbed)[0]
    stabilized_weight = minimum_variance_two_asset(perturbed, float(oracle["ridge"]))[0]
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
    var, es = empirical_var_es(
        np.asarray(oracle["losses"], dtype=float),
        float(oracle["confidence"]),
        minimum_tail_observations=1,
    )
    stress_loss = float(np.asarray(oracle["stress_shocks"], dtype=float) @ weights * -1.0)
    failures = (
        expect_rejection(lambda: validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]])), "positive semidefinite"),
        expect_rejection(
            lambda: cost_aware_rebalance(
                np.array([0.08, 0.04]), direct, np.array([0.7, 0.3]), risk_aversion=1.0,
                cost_rate=0.001, capital=100000.0, maximum_weight=0.6,
                tradable=np.array([0, 1]), grid_step=0.1,
            ),
            "no feasible portfolio",
        ),
        expect_rejection(
            lambda: empirical_var_es(np.array([1.0, 2.0, 3.0]), 0.95),
            "effective tail observations",
        ),
    )
    observed: dict[str, float | int] = {
        "variance_1": direct[0, 0], "covariance_12": direct[0, 1], "variance_2": direct[1, 1],
        "minimum_weight_1": minimum[0], "minimum_weight_2": minimum[1],
        "ill_weight": ill_weight, "perturbed_weight": perturbed_weight, "stabilized_weight": stabilized_weight,
        "rebalance_weight_1": weights[0], "rebalance_weight_2": weights[1],
        "turnover": turnover, "cost_cash": cost_cash, "var": var, "es": es, "stress_loss": stress_loss,
        "confidence": float(oracle["confidence"]), "ridge": float(oracle["ridge"]),
        "risk_aversion": float(oracle["risk_aversion"]), "cost_rate": float(oracle["cost_rate"]),
        "maximum_weight": float(oracle["maximum_weight"]), "grid_step": float(oracle["grid_step"]),
        "tradable_assets": int(np.asarray(oracle["tradable"], dtype=int).sum()),
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
        f"weights=({minimum[0]:.6f},{minimum[1]:.6f}) "
        f"stability=({ill_weight:.6f},{perturbed_weight:.6f},{stabilized_weight:.6f}) "
        f"rebalance=({weights[0]:.6f},{weights[1]:.6f},{turnover:.6f},{cost_cash:.6f}) "
        f"risk=({var:.6f},{es:.6f},{stress_loss:.6f}) failures=({','.join(str(value) for value in failures)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch05/oracle.json")))
