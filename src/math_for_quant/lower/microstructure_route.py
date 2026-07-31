from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from math_for_quant.lower.microstructure_control import (
    execution_path,
    market_making_cycle,
    optimal_execution,
)
from math_for_quant.lower.microstructure_events import (
    OrderBook, hawkes_log_likelihood,
    joint_order_price_beta,
    poisson_mle,
    queue_fill_probability,
    seasonally_adjusted_poisson_mle,
)
from math_for_quant.lower.microstructure_simulation import (
    analyze_sec_order_placement,
    paired_event_simulation,
)
from math_for_quant.lower.microstructure_events_library import (
    library_hawkes_log_likelihood, library_joint_beta, library_poisson_mle,
    library_queue_fill_probability, library_seasonal_poisson_mle,
)
from math_for_quant.lower.microstructure_control_library import (
    closed_form_inventory_quotes, enumerate_execution,
)
from math_for_quant.lower.microstructure_simulation_library import (
    independent_inventory_feedback_ledger,
    vectorized_static_pnl,
)


ROOT = Path(__file__).resolve().parents[3]


def run_events(fixture: dict[str, Any]) -> dict[str, float | int]:
    intensity, log_likelihood = poisson_mle(fixture["interarrivals"])
    fill_probability = queue_fill_probability(
        queue_ahead=int(fixture["queue_ahead"]), own_quantity=int(fixture["own_quantity"]),
        depletion_intensity=float(fixture["depletion_intensity"]), horizon=float(fixture["horizon"]),
    )
    beta = joint_order_price_beta(
        np.asarray(fixture["signs"], dtype=float), np.asarray(fixture["price_changes"], dtype=float)
    )
    hawkes_times = np.asarray(fixture["hawkes_times"], dtype=float)
    hawkes = hawkes_log_likelihood(
        hawkes_times,
        float(fixture["hawkes_baseline"]),
        float(fixture["hawkes_alpha"]),
        float(fixture["hawkes_beta"]),
        horizon=float(fixture["hawkes_horizon"]),
        initial_excitation=float(fixture["hawkes_initial_excitation"]),
    )
    seasonal_intensity, seasonal_residuals = seasonally_adjusted_poisson_mle(
        fixture["seasonal_interarrivals"], fixture["seasonal_multipliers"]
    )
    book = OrderBook()
    book.add("b1", "buy", 100.0, 3)
    book.add("b2", "buy", 100.0, 2)
    queue_ahead = book.queue_ahead("b2")
    fills, unfilled = book.market("sell", 4, allow_partial=True)
    book.assert_invariants()

    real = analyze_sec_order_placement(ROOT / str(fixture["real_snapshot"]))
    library_intensity = library_poisson_mle(fixture["interarrivals"])
    library_probability = library_queue_fill_probability(
        int(fixture["queue_ahead"]) + int(fixture["own_quantity"]),
        float(fixture["depletion_intensity"]) * float(fixture["horizon"]),
    )
    library_beta = library_joint_beta(
        np.asarray(fixture["signs"], dtype=float), np.asarray(fixture["price_changes"], dtype=float)
    )
    library_hawkes = library_hawkes_log_likelihood(
        hawkes_times,
        float(fixture["hawkes_baseline"]), float(fixture["hawkes_alpha"]),
        float(fixture["hawkes_beta"]), float(fixture["hawkes_horizon"]),
        float(fixture["hawkes_initial_excitation"]),
    )
    library_seasonal, library_residuals = library_seasonal_poisson_mle(
        fixture["seasonal_interarrivals"], fixture["seasonal_multipliers"]
    )
    real_depletion_event_rate = float(fixture["real_depletion_event_rate"])
    if (
        not np.isfinite(real_depletion_event_rate)
        or real_depletion_event_rate <= 0.0
        or fixture.get("real_depletion_event_rate_unit") != "terminal_events_per_second"
    ):
        raise ValueError("real depletion event-rate assumption is invalid")
    real_fill = queue_fill_probability(
        queue_ahead=int(fixture["queue_ahead"]), own_quantity=int(fixture["own_quantity"]),
        depletion_intensity=(
            real_depletion_event_rate * float(real["implied_execution_probability"])
        ),
        horizon=float(fixture["horizon"]),
    )
    return {
        "poisson_intensity": intensity, "poisson_log_likelihood": log_likelihood,
        "queue_fill_probability": fill_probability, "joint_beta": beta,
        "queue_ahead": queue_ahead, "first_fill_quantity": fills[0][1],
        "partial_fill_quantity": fills[1][1], "unfilled": unfilled,
        "hawkes_log_likelihood": hawkes,
        "hawkes_branching_ratio": float(fixture["hawkes_alpha"]) / float(fixture["hawkes_beta"]),
        "seasonal_intensity": seasonal_intensity,
        "seasonal_residual_mean": float(np.mean(seasonal_residuals)),
        "real_implied_execution_probability": real["implied_execution_probability"],
        "real_depletion_event_rate": real_depletion_event_rate,
        "real_queue_fill_probability": real_fill,
        "event_library_gap": max(
            abs(intensity - library_intensity), abs(fill_probability - library_probability),
            abs(beta - library_beta), abs(hawkes - library_hawkes),
            abs(seasonal_intensity - library_seasonal),
            float(np.max(np.abs(seasonal_residuals - library_residuals))),
        ),
    }


def run_control(fixture: dict[str, Any]) -> dict[str, float | int | str]:
    real = analyze_sec_order_placement(ROOT / str(fixture["real_snapshot"]))
    execution = execution_path(
        schedule=[int(x) for x in fixture["schedule"]],
        fill_rates=[float(x) for x in fixture["fill_rates"]],
        initial_inventory=int(fixture["initial_inventory"]), arrival_price=float(fixture["arrival_price"]),
        temporary_impact=float(fixture["temporary_impact"]), permanent_impact=float(fixture["permanent_impact"]),
        stop_after=int(fixture["stop_after"]),
    )
    schedule, cost = optimal_execution(
        inventory=int(fixture["dp_inventory"]), steps=int(fixture["dp_steps"]),
        temporary_impact=float(fixture["dp_temporary_impact"]),
        permanent_impact=float(fixture["dp_permanent_impact"]),
        inventory_risk=float(fixture["dp_inventory_risk"]), maximum_slice=int(fixture["dp_maximum_slice"]),
    )
    maker = market_making_cycle(
        midprices=[float(x) for x in fixture["midprices"]], fills=list(fixture["fills"]), initial_inventory=0,
        half_spread=float(fixture["half_spread"]), inventory_skew=float(fixture["inventory_skew"]),
    )
    real_execution = execution_path(
        schedule=[int(x) for x in fixture["schedule"]],
        fill_rates=[
            float(real["inside_execution_probability"]),
            float(real["at_spread_execution_probability"]),
            float(real["within_50bp_execution_probability"]),
        ],
        initial_inventory=int(fixture["initial_inventory"]),
        arrival_price=float(fixture["arrival_price"]),
        temporary_impact=float(fixture["temporary_impact"]),
        permanent_impact=float(fixture["permanent_impact"]),
        stop_after=int(fixture["stop_after"]),
    )
    enumerated_schedule, enumerated_cost = enumerate_execution(
        inventory=int(fixture["dp_inventory"]), steps=int(fixture["dp_steps"]),
        temporary_impact=float(fixture["dp_temporary_impact"]),
        permanent_impact=float(fixture["dp_permanent_impact"]),
        inventory_risk=float(fixture["dp_inventory_risk"]), maximum_slice=int(fixture["dp_maximum_slice"]),
    )
    library_bid, library_ask = closed_form_inventory_quotes(
        float(fixture["midprices"][-1]), maker.inventories[1],
        float(fixture["half_spread"]), float(fixture["inventory_skew"]),
    )
    return {
        "execution_filled": sum(execution.filled),
        "execution_remaining": execution.remaining,
        "execution_shortfall": execution.implementation_shortfall,
        "optimal_schedule": "-".join(str(value) for value in schedule),
        "optimal_cost": cost,
        "maker_inventory": maker.inventories[-1],
        "maker_next_bid": maker.bids[1],
        "maker_next_ask": maker.asks[1],
        "real_execution_filled": sum(real_execution.filled),
        "real_execution_remaining": real_execution.remaining,
        "real_execution_shortfall": real_execution.implementation_shortfall,
        "control_library_gap": max(
            abs(cost - enumerated_cost),
            0.0 if schedule == enumerated_schedule else 1.0,
            abs(maker.bids[1] - library_bid), abs(maker.asks[1] - library_ask),
        ),
    }


def run_simulation(fixture: dict[str, Any]) -> dict[str, float | int | str]:
    real = analyze_sec_order_placement(ROOT / str(fixture["real_snapshot"]))
    simulation = paired_event_simulation(
        seed=int(fixture["seed"]), events=int(fixture["events"]),
        base_intensity=float(fixture["base_intensity"]),
        fill_probability=float(real["implied_execution_probability"]),
    )
    rng = np.random.default_rng(int(fixture["seed"]))
    interarrivals = rng.exponential(1.0 / float(fixture["base_intensity"]), int(fixture["events"]))
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(fixture["events"]))
    uniforms = rng.random(int(fixture["events"]))
    fills = uniforms < float(real["implied_execution_probability"])
    innovations = rng.normal(0.0, 0.02, int(fixture["events"]))
    changes = 0.006 * signs + innovations * np.sqrt(interarrivals)
    library_pnl = vectorized_static_pnl(signs, changes, 0.01, fills)
    control_library = independent_inventory_feedback_ledger(
        signs, changes, uniforms, float(real["implied_execution_probability"])
    )
    control_gap = max(
        abs(simulation.control_pnl - control_library[0]),
        abs(simulation.control_fills - control_library[1]),
        abs(simulation.control_ending_inventory - control_library[2]),
        abs(simulation.control_max_abs_inventory - control_library[3]),
    )
    return {
        "simulation_events": simulation.events,
        "baseline_fills": simulation.baseline_fills,
        "control_fills": simulation.control_fills,
        "baseline_pnl": simulation.baseline_pnl,
        "control_pnl": simulation.control_pnl,
        "baseline_ending_inventory": simulation.baseline_ending_inventory,
        "control_ending_inventory": simulation.control_ending_inventory,
        "baseline_max_abs_inventory": simulation.baseline_max_abs_inventory,
        "control_max_abs_inventory": simulation.control_max_abs_inventory,
        "simulation_library_gap": max(
            abs(simulation.baseline_pnl - library_pnl), control_gap
        ),
        "real_categories": real["categories"],
        "real_weighted_cancel_to_trade": real["weighted_cancel_to_trade"],
        "real_implied_execution_probability": real["implied_execution_probability"],
        "real_maximum_cancel_to_trade": real["maximum_cancel_to_trade"],
    }


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "data" / "fixtures" / name).read_text(encoding="utf-8"))


def build_route_results() -> dict[str, float | int | str]:
    return {
        **run_events(_fixture("microstructure-events.json")),
        **run_control(_fixture("microstructure-control.json")),
        **run_simulation(_fixture("microstructure-simulation.json")),
    }


def render_route_report(result: dict[str, float | int | str]) -> str:
    return f"""# 高频、微观结构与执行 v0.3 路线报告

- 事件模型：四段等待时间的 Poisson MLE 为 {result['poisson_intensity']:.6f}，对数似然 {result['poisson_log_likelihood']:.6f}；排在 2 手之后、1 秒内由强度 3 的耗尽流完全成交概率为 {result['queue_fill_probability']:.6f}；订单方向与价格变化的手算 OLS 斜率为 {result['joint_beta']:.6f}，独立数值实现最大差 {result['event_library_gap']:.3e}。
- 自激与季节性：显式观察窗上的 Hawkes 对数似然为 {result['hawkes_log_likelihood']:.6f}，分支比 {result['hawkes_branching_ratio']:.6f}；季节调整 Poisson 基准强度 {result['seasonal_intensity']:.6f}，变换后残差均值 {result['seasonal_residual_mean']:.6f}。稳定性、观察窗和季节暴露均进入可执行证据。
- 订单簿：第二张买单前方数量 {result['queue_ahead']}；4 手市价卖单先成交 {result['first_fill_quantity']} 手，再部分成交 {result['partial_fill_quantity']} 手，未成交 {result['unfilled']}；价格优先、时间优先和非负数量不变量均通过。
- 执行控制：计划 9 手实际成交 {result['execution_filled']}、剩余 {result['execution_remaining']}，含临时与永久冲击的 implementation shortfall 为 {result['execution_shortfall']:.6f}；动态规划最优日程 {result['optimal_schedule']}，目标 {result['optimal_cost']:.6f}，完整枚举与闭式报价最大差 {result['control_library_gap']:.3e}。
- 做市反馈：首次 bid 成交后库存为 {result['maker_inventory']}，下一轮双边报价下移至 {result['maker_next_bid']:.6f}/{result['maker_next_ask']:.6f}，库存而非事后文字真正进入报价函数。
- 配对仿真：{result['simulation_events']} 个事件共享同一随机订单方向和成交均匀数；静态/库存反馈报价分别成交 {result['baseline_fills']}/{result['control_fills']} 次，现金加期末库存盯市 PnL 为 {result['baseline_pnl']:.6f}/{result['control_pnl']:.6f}，期末库存 {result['baseline_ending_inventory']}/{result['control_ending_inventory']}，最大绝对库存 {result['baseline_max_abs_inventory']}/{result['control_max_abs_inventory']}，独立双账本最大差 {result['simulation_library_gap']:.3e}。报价通过各自阈值改变成交，但两策略共享随机源；这是配对比较，不是盈利证明。
- 真实数据轨：SEC 公共领域订单位置摘要含 {result['real_categories']} 个互斥类别，加权 cancel-to-trade 比率 {result['real_weighted_cancel_to_trade']:.6f}，隐含执行概率 {result['real_implied_execution_probability']:.6f}，最大类别比率 {result['real_maximum_cancel_to_trade']:.6f}。另行声明的终止事件率压力为 {result['real_depletion_event_rate']:.6f} 次/秒；两者共同驱动队列压力（完全成交概率 {result['real_queue_fill_probability']:.6f}）、执行路径（成交 {result['real_execution_filled']}、剩余 {result['real_execution_remaining']}、shortfall {result['real_execution_shortfall']:.6f}）和仿真成交阈值。
- 限制：SEC 摘要是聚合的历史位置统计，不含逐档深度、订单方向、队列标识或延迟；由 cancel-to-trade 比率换算的执行概率只适合压力实验。合成订单簿、执行与做市实验只验证状态转移、成本恒等式和比较协议，不能外推成交质量或策略收益。
"""


def build_route_report() -> str:
    return render_route_report(build_route_results())


__all__ = ["build_route_report", "build_route_results", "render_route_report", "run_control", "run_events", "run_simulation"]
