from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from math_for_quant.evidence import load_oracle_bundle


def poisson_mle(interarrivals: list[float]) -> tuple[float, float]:
    if not interarrivals or any(not math.isfinite(value) or value <= 0.0 for value in interarrivals):
        raise ValueError("interarrival times must be positive")
    total = float(sum(interarrivals))
    intensity = len(interarrivals) / total
    log_likelihood = len(interarrivals) * math.log(intensity) - intensity * total
    return intensity, log_likelihood


def validate_constant_intensity(window_counts: list[int], maximum_ratio: float) -> None:
    if not math.isfinite(maximum_ratio) or maximum_ratio <= 0.0:
        raise ValueError("maximum seasonality ratio must be positive and finite")
    if not window_counts or min(window_counts) <= 0:
        raise ValueError("window counts must be positive")
    if max(window_counts) / min(window_counts) > maximum_ratio:
        raise ValueError("intraday seasonality invalidates constant intensity")


def validate_hawkes_stability(alpha: float, beta: float) -> None:
    if not math.isfinite(alpha) or not math.isfinite(beta) or alpha < 0.0 or beta <= 0.0 or alpha / beta >= 1.0:
        raise ValueError("unstable Hawkes branching ratio")


@dataclass
class Order:
    order_id: str
    side: str
    price: float
    quantity: int
    sequence: int


class OrderBook:
    def __init__(self) -> None:
        self.orders: list[Order] = []
        self.sequence = 0
        self.traded_quantity = 0

    def add(self, order_id: str, side: str, price: float, quantity: int) -> None:
        if side not in {"buy", "sell"} or not math.isfinite(price) or price <= 0.0 or quantity <= 0:
            raise ValueError("invalid limit order")
        if any(order.order_id == order_id for order in self.orders):
            raise ValueError("duplicate order id")
        opposite = [order for order in self.orders if order.side != side]
        if opposite:
            opposing_best = min(order.price for order in opposite) if side == "buy" else max(order.price for order in opposite)
            if (side == "buy" and price >= opposing_best) or (side == "sell" and price <= opposing_best):
                raise ValueError("crossed limit order requires aggressive execution")
        self.sequence += 1
        self.orders.append(Order(order_id, side, price, quantity, self.sequence))

    def cancel(self, order_id: str, quantity: int) -> None:
        order = next((item for item in self.orders if item.order_id == order_id), None)
        if order is None or quantity <= 0 or quantity > order.quantity:
            raise ValueError("cancel exceeds live order quantity")
        order.quantity -= quantity
        if order.quantity == 0:
            self.orders.remove(order)

    def market(self, side: str, quantity: int) -> list[tuple[str, int, float]]:
        if side not in {"buy", "sell"} or quantity <= 0:
            raise ValueError("invalid market order")
        passive_side = "sell" if side == "buy" else "buy"
        candidates = [order for order in self.orders if order.side == passive_side]
        candidates.sort(key=lambda order: (order.price if side == "buy" else -order.price, order.sequence))
        if sum(order.quantity for order in candidates) < quantity:
            raise ValueError("insufficient displayed liquidity")
        fills: list[tuple[str, int, float]] = []
        remaining = quantity
        for order in candidates:
            if remaining == 0:
                break
            fill = min(order.quantity, remaining)
            order.quantity -= fill
            remaining -= fill
            self.traded_quantity += fill
            fills.append((order.order_id, fill, order.price))
            if order.quantity == 0:
                self.orders.remove(order)
        return fills

    def best(self, side: str) -> tuple[float, int]:
        levels = [order for order in self.orders if order.side == side]
        if not levels:
            raise ValueError("book side is empty")
        price = max(order.price for order in levels) if side == "buy" else min(order.price for order in levels)
        return price, sum(order.quantity for order in levels if order.price == price)


def replay(events: list[dict[str, object]]) -> tuple[OrderBook, list[tuple[str, int, float]]]:
    book = OrderBook()
    fills: list[tuple[str, int, float]] = []
    for event in events:
        kind = str(event["type"])
        if kind == "add":
            book.add(str(event["order_id"]), str(event["side"]), float(event["price"]), int(event["quantity"]))
        elif kind == "cancel":
            book.cancel(str(event["order_id"]), int(event["quantity"]))
        elif kind == "market":
            fills.extend(book.market(str(event["side"]), int(event["quantity"])))
        else:
            raise ValueError("unknown book event")
    return book, fills


def twap_schedule(inventory: int, steps: int) -> list[int]:
    if inventory < 0 or steps <= 0:
        raise ValueError("inventory and steps must be nonnegative")
    base, extra = divmod(inventory, steps)
    return [base + (1 if index < extra else 0) for index in range(steps)]


def optimal_execution(inventory: int, steps: int, impact: float, inventory_risk: float, max_slice: int, displayed_liquidity: int) -> tuple[list[int], float]:
    if not math.isfinite(impact) or impact <= 0.0:
        raise ValueError("temporary impact must be positive")
    if not math.isfinite(inventory_risk) or inventory_risk < 0.0:
        raise ValueError("inventory risk must be finite and nonnegative")
    if max_slice <= 0 or displayed_liquidity <= 0 or max_slice > displayed_liquidity:
        raise ValueError("execution must respect finite displayed liquidity")

    @lru_cache(maxsize=None)
    def solve(step: int, remaining: int) -> tuple[float, tuple[int, ...]]:
        if step == steps:
            return (0.0, ()) if remaining == 0 else (float("inf"), ())
        best_cost, best_path = float("inf"), ()
        steps_left = steps - step - 1
        lower = max(0, remaining - steps_left * max_slice)
        for quantity in range(lower, min(max_slice, remaining) + 1):
            after = remaining - quantity
            future_cost, future_path = solve(step + 1, after)
            cost = impact * quantity * quantity + inventory_risk * after * after + future_cost
            if cost < best_cost - 1e-12:
                best_cost, best_path = cost, (quantity,) + future_path
        return best_cost, best_path

    cost, schedule = solve(0, inventory)
    if not math.isfinite(cost):
        raise ValueError("inventory cannot be completed within liquidity limits")
    return list(schedule), cost


def validate_information_time(decision_time: float, signal_time: float) -> None:
    if not math.isfinite(decision_time) or not math.isfinite(signal_time):
        raise ValueError("information timestamps must be finite")
    if signal_time > decision_time:
        raise ValueError("future price is not observable at decision time")


def inventory_path(inventory: int, schedule: list[int]) -> list[int]:
    path = [inventory]
    for quantity in schedule:
        path.append(path[-1] - quantity)
    if path[-1] != 0 or min(path) < 0:
        raise ValueError("execution schedule violates terminal inventory")
    return path


def execution_objective(schedule: list[int], inventory: int, impact: float, inventory_risk: float) -> tuple[float, float, float]:
    path = inventory_path(inventory, schedule)
    impact_cost = impact * sum(quantity * quantity for quantity in schedule)
    risk_cost = inventory_risk * sum(level * level for level in path[1:])
    return impact_cost, risk_cost, impact_cost + risk_cost


def latency_budget(parts: list[float]) -> float:
    if any(not math.isfinite(value) or value < 0.0 for value in parts):
        raise ValueError("latency components must be finite and nonnegative")
    return float(sum(parts))


def expect_rejection(action, diagnostic: str) -> int:
    try:
        action()
    except ValueError as error:
        if diagnostic not in str(error):
            raise AssertionError(f"wrong diagnostic: {error}") from error
        return 1
    raise AssertionError(f"expected rejection containing {diagnostic!r}")


def render_execution_contract(observed: dict[str, object]) -> str:
    return (
        f"hawkes=({float(observed['hawkes_alpha']):.4f},{float(observed['hawkes_beta']):.4f})，"
        f"true_intensity={float(observed['true_intensity']):.4f}，inventory={int(observed['inventory'])}，"
        f"steps={int(observed['steps'])}，impact={float(observed['impact']):.4f}，"
        f"inventory_risk={float(observed['inventory_risk']):.4f}，max_slice={int(observed['max_slice'])}，"
        f"displayed_liquidity={int(observed['displayed_liquidity'])}，total_latency_ms={float(observed['total_latency_ms']):.3f}"
    )


def render_report(observed: dict[str, object]) -> str:
    contract = render_execution_contract(observed)
    return f"""# 高频、微观结构与执行可复现研究包

- 订单流：手算强度 {observed['intensity']:.6f}，Poisson 对数似然 {observed['log_likelihood']:.6f}；Hawkes 分枝比小于 1 的稳定条件已检查。
- 仿真：seed={int(observed['seed'])}，事件数 {int(observed['simulation_events'])}，到达间隔均值 {observed['simulated_mean']:.6f}、方差 {observed['simulated_variance']:.6f}，均值 95% 区间 [{observed['ci_low']:.6f}, {observed['ci_high']:.6f}] 覆盖理论均值 {observed['theoretical_mean']:.6f}。
- 订单簿：最优买价 {observed['best_bid']:.2f} 数量 {int(observed['bid_quantity'])}；最优卖价 {observed['best_ask']:.2f} 数量 {int(observed['ask_quantity'])}；累计成交 {int(observed['traded_quantity'])}。同价 FIFO、撤单与部分成交均进入回放。
- 执行：TWAP 为 ({observed['twap_schedule']})，冲击/库存/总目标为 {float(observed['twap_impact_cost']):.6f}/{float(observed['twap_risk_cost']):.6f}/{float(observed['twap_total_cost']):.6f}；最优序列为 ({observed['optimal_schedule']})，冲击/库存/总目标为 {float(observed['optimal_impact_cost']):.6f}/{float(observed['optimal_risk_cost']):.6f}/{float(observed['optimal_cost']):.6f}。
- 库存：最优路径为 ({observed['inventory_path']})，峰值 {int(observed['peak_inventory'])}，终值 {int(observed['final_inventory'])}。实验契约：{contract}。
- 失败边界：日内季节性、Hawkes 不稳定、错误撤单、穿价限价、零冲击、无限流动性与未来价格可见分别拒绝。
- 限制：合成事件没有真实延迟、隐藏单、撮合所制度差异、冲击反馈或排队位置不确定性；本结果不可声称生产做市或执行能力。
- 复现命令：`uv run python notebooks/lower/ch06_microstructure.py evidence/lower-ch06/oracle.json`。
"""


def main(oracle_path: Path = Path("evidence/lower-ch06/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    intensity, log_likelihood = poisson_mle([float(value) for value in oracle["interarrivals"]])
    validate_hawkes_stability(float(oracle["hawkes_alpha"]), float(oracle["hawkes_beta"]))
    rng = np.random.default_rng(int(oracle["seed"]))
    simulated = rng.exponential(1.0 / float(oracle["true_intensity"]), int(oracle["simulation_events"]))
    simulated_mean = float(simulated.mean())
    simulated_variance = float(simulated.var(ddof=1))
    standard_error = math.sqrt(simulated_variance / simulated.size)
    ci_low, ci_high = simulated_mean - 1.96 * standard_error, simulated_mean + 1.96 * standard_error
    theoretical_mean = 1.0 / float(oracle["true_intensity"])
    simulation_covers = int(ci_low <= theoretical_mean <= ci_high)

    book, fills = replay(list(oracle["events"]))
    best_bid, bid_quantity = book.best("buy")
    best_ask, ask_quantity = book.best("sell")
    expected_fill_ids = [str(value) for value in oracle["expected_fill_ids"]]
    if [fill[0] for fill in fills] != expected_fill_ids:
        raise SystemExit("FIFO fill ledger drifted")

    inventory, steps = int(oracle["inventory"]), int(oracle["steps"])
    twap = twap_schedule(inventory, steps)
    impact = float(oracle["impact"])
    inventory_risk = float(oracle["inventory_risk"])
    twap_impact_cost, twap_risk_cost, twap_total_cost = execution_objective(twap, inventory, impact, inventory_risk)
    optimal, optimal_cost = optimal_execution(
        inventory, steps, impact, inventory_risk,
        int(oracle["max_slice"]), int(oracle["displayed_liquidity"]),
    )
    optimal_impact_cost, optimal_risk_cost, recomputed_optimal_cost = execution_objective(optimal, inventory, impact, inventory_risk)
    if abs(optimal_cost - recomputed_optimal_cost) > 1e-12:
        raise SystemExit("execution objective decomposition drifted")
    optimal_inventory_path = inventory_path(inventory, optimal)
    total_latency_ms = latency_budget([float(value) for value in oracle["latency_ms"]])
    failures = (
        expect_rejection(lambda: validate_constant_intensity([8, 2], 2.0), "intraday seasonality"),
        expect_rejection(lambda: validate_hawkes_stability(1.0, 1.0), "unstable Hawkes"),
        expect_rejection(lambda: replay([{"type": "add", "order_id": "x", "side": "buy", "price": 1, "quantity": 1}, {"type": "cancel", "order_id": "x", "quantity": 2}]), "cancel exceeds"),
        expect_rejection(lambda: replay([{"type": "add", "order_id": "a", "side": "sell", "price": 101, "quantity": 1}, {"type": "add", "order_id": "b", "side": "buy", "price": 101, "quantity": 1}]), "crossed limit order"),
        expect_rejection(lambda: optimal_execution(6, 3, 0.0, 0.3, 3, 3), "impact must be positive"),
        expect_rejection(lambda: optimal_execution(6, 3, 1.0, 0.3, 7, 3), "finite displayed liquidity"),
        expect_rejection(lambda: validate_information_time(10.0, 11.0), "future price"),
    )
    observed: dict[str, object] = {
        "intensity": intensity, "log_likelihood": log_likelihood, "simulation_covers": simulation_covers,
        "seed": int(oracle["seed"]), "simulation_events": int(oracle["simulation_events"]),
        "simulated_mean": simulated_mean, "simulated_variance": simulated_variance,
        "ci_low": ci_low, "ci_high": ci_high, "theoretical_mean": theoretical_mean,
        "best_bid": best_bid, "bid_quantity": bid_quantity, "best_ask": best_ask, "ask_quantity": ask_quantity,
        "traded_quantity": book.traded_quantity,
        "twap_schedule": ", ".join(str(value) for value in twap),
        "twap_impact_cost": twap_impact_cost, "twap_risk_cost": twap_risk_cost, "twap_total_cost": twap_total_cost,
        "optimal_schedule": ", ".join(str(value) for value in optimal),
        "optimal_impact_cost": optimal_impact_cost, "optimal_risk_cost": optimal_risk_cost, "optimal_cost": optimal_cost,
        "inventory_path": ", ".join(str(value) for value in optimal_inventory_path),
        "peak_inventory": max(optimal_inventory_path), "final_inventory": optimal_inventory_path[-1],
        "hawkes_alpha": float(oracle["hawkes_alpha"]), "hawkes_beta": float(oracle["hawkes_beta"]),
        "true_intensity": float(oracle["true_intensity"]), "inventory": inventory, "steps": steps,
        "impact": impact, "inventory_risk": inventory_risk, "max_slice": int(oracle["max_slice"]),
        "displayed_liquidity": int(oracle["displayed_liquidity"]), "total_latency_ms": total_latency_ms,
        **{f"failure_{index + 1}": value for index, value in enumerate(failures)},
    }
    tolerance = float(oracle["absolute_tolerance"])
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected:
                raise SystemExit(f"{name} failed: {value} != {expected}")
        elif not math.isfinite(float(value)) or abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} failed: {value} != {expected}")
    report_path = Path(oracle["report"])
    if report_path.read_text(encoding="utf-8") != render_report(observed):
        raise SystemExit(f"reproducible report drifted: {report_path}")
    print(
        f"oracle=passed point=({intensity:.6f},{log_likelihood:.6f},{simulation_covers}) "
        f"book=({best_bid:.6f},{bid_quantity},{best_ask:.6f},{ask_quantity},{book.traded_quantity}) "
        f"execution=({','.join(str(value) for value in twap)},{','.join(str(value) for value in optimal)},"
        f"{twap_impact_cost:.6f},{optimal_cost:.6f}) failures=({','.join(str(value) for value in failures)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch06/oracle.json")))
