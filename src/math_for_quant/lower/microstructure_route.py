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
    OrderBook,
    joint_order_price_beta,
    poisson_mle,
    queue_fill_probability,
)
from math_for_quant.lower.microstructure_simulation import (
    analyze_coinbase_trades,
    paired_event_simulation,
)
from math_for_quant.lower.microstructure_events_library import (
    library_joint_beta, library_poisson_mle, library_queue_fill_probability,
)
from math_for_quant.lower.microstructure_control_library import (
    closed_form_inventory_quotes, enumerate_execution,
)
from math_for_quant.lower.microstructure_simulation_library import vectorized_static_pnl


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
    book = OrderBook()
    book.add("b1", "buy", 100.0, 3)
    book.add("b2", "buy", 100.0, 2)
    queue_ahead = book.queue_ahead("b2")
    fills, unfilled = book.market("sell", 4, allow_partial=True)
    book.assert_invariants()

    real = analyze_coinbase_trades(ROOT / str(fixture["real_snapshot"]))
    library_intensity = library_poisson_mle(fixture["interarrivals"])
    library_probability = library_queue_fill_probability(
        int(fixture["queue_ahead"]) + int(fixture["own_quantity"]),
        float(fixture["depletion_intensity"]) * float(fixture["horizon"]),
    )
    library_beta = library_joint_beta(
        np.asarray(fixture["signs"], dtype=float), np.asarray(fixture["price_changes"], dtype=float)
    )
    return {
        "poisson_intensity": intensity, "poisson_log_likelihood": log_likelihood,
        "queue_fill_probability": fill_probability, "joint_beta": beta,
        "queue_ahead": queue_ahead, "first_fill_quantity": fills[0][1],
        "partial_fill_quantity": fills[1][1], "unfilled": unfilled,
        "real_event_duration": real["duration_seconds"],
        "real_event_beta": real["order_price_beta"],
        "event_library_gap": max(
            abs(intensity - library_intensity), abs(fill_probability - library_probability), abs(beta - library_beta)
        ),
    }


def run_control(fixture: dict[str, Any]) -> dict[str, float | int | str]:
    real_payload = json.loads((ROOT / str(fixture["real_snapshot"])).read_text(encoding="utf-8"))
    real_rows = real_payload["trades"]
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
        "real_arrival_price": float(real_rows[0]["price"]),
        "real_total_size": sum(float(row["size"]) for row in real_rows),
        "real_max_trade_size": max(float(row["size"]) for row in real_rows),
        "control_library_gap": max(
            abs(cost - enumerated_cost),
            0.0 if schedule == enumerated_schedule else 1.0,
            abs(maker.bids[1] - library_bid), abs(maker.asks[1] - library_ask),
        ),
    }


def run_simulation(fixture: dict[str, Any]) -> dict[str, float | int | str]:
    simulation = paired_event_simulation(
        seed=int(fixture["seed"]), events=int(fixture["events"]),
        base_intensity=float(fixture["base_intensity"]),
    )
    real = analyze_coinbase_trades(ROOT / str(fixture["real_snapshot"]))
    rng = np.random.default_rng(int(fixture["seed"]))
    interarrivals = rng.exponential(1.0 / float(fixture["base_intensity"]), int(fixture["events"]))
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(fixture["events"]))
    innovations = rng.normal(0.0, 0.02, int(fixture["events"]))
    changes = 0.006 * signs + innovations * np.sqrt(interarrivals)
    library_pnl = vectorized_static_pnl(signs, changes, 0.01)
    return {
        "simulation_events": simulation.events, "baseline_pnl": simulation.baseline_pnl,
        "control_pnl": simulation.control_pnl, "ending_inventory": simulation.ending_inventory,
        "simulation_library_gap": abs(simulation.baseline_pnl - library_pnl),
        **{f"real_{key}": value for key, value in real.items()},
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
- 订单簿：第二张买单前方数量 {result['queue_ahead']}；4 手市价卖单先成交 {result['first_fill_quantity']} 手，再部分成交 {result['partial_fill_quantity']} 手，未成交 {result['unfilled']}；价格优先、时间优先和非负数量不变量均通过。
- 执行控制：计划 9 手实际成交 {result['execution_filled']}、剩余 {result['execution_remaining']}，含临时与永久冲击的 implementation shortfall 为 {result['execution_shortfall']:.6f}；动态规划最优日程 {result['optimal_schedule']}，目标 {result['optimal_cost']:.6f}，完整枚举与闭式报价最大差 {result['control_library_gap']:.3e}。
- 做市反馈：首次 bid 成交后库存为 {result['maker_inventory']}，下一轮双边报价下移至 {result['maker_next_bid']:.6f}/{result['maker_next_ask']:.6f}，库存而非事后文字真正进入报价函数。
- 配对仿真：{result['simulation_events']} 个事件共享同一随机订单方向；静态与库存控制规则 PnL 分别为 {result['baseline_pnl']:.6f}/{result['control_pnl']:.6f}，控制规则期末库存 {result['ending_inventory']}，向量化静态账本差 {result['simulation_library_gap']:.3e}。这是共同随机数比较，不是盈利证明。
- 真实数据轨：冻结 Coinbase BTC-USD 快照覆盖 trade id {result['real_first_trade_id']}--{result['real_last_trade_id']}，共 {result['real_trades']} 笔、{result['real_duration_seconds']:.6f} 秒，maker-sell 比例 {result['real_maker_sell_share']:.6f}，订单方向--价格变化斜率 {result['real_order_price_beta']:.6f}。`side` 按官方文档解释为 maker side。
- 限制：公开成交快照不含逐档深度、撤单、队列位置或延迟；合成订单簿、执行与做市实验只验证状态转移、成本恒等式和比较协议，不能外推成交质量或策略收益。
"""


def build_route_report() -> str:
    return render_route_report(build_route_results())


__all__ = ["build_route_report", "build_route_results", "render_route_report", "run_control", "run_events", "run_simulation"]
