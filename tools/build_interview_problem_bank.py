from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


# The catalog follows every named entry in the source book's contents.  The
# user-approved denominator is the complete frozen catalog, including entries
# that introduce a technique rather than carrying a puzzle-style title.
GROUPS = [
    (2, "2.1", 3, "problem", "Screwy pirates|Tiger and sheep"),
    (2, "2.2", 5, "problem", "River crossing|Birthday problem|Card game|Burning ropes|Defective ball|Trailing zeros|Horse race|Infinite sequence"),
    (2, "2.3", 10, "problem", "Box packing|Calendar cubes|Door to offer|Message delivery|Last ball|Light switches|Quant salary"),
    (2, "2.4", 15, "problem", "Coin piles|Mislabeled bags|Wise men"),
    (2, "2.5", 17, "problem", "Clock pieces|Missing integers|Counterfeit coins I"),
    (2, "2.6", 20, "problem", "Matching socks|Handshakes|Have we met before?|Ants on a square|Counterfeit coins II"),
    (2, "2.7", 23, "problem", "Prisoner problem|Division by 9|Chameleon colors"),
    (2, "2.8", 27, "problem", "Coin split problem|Chocolate bar problem|Race track"),
    (2, "2.9", 31, "problem", "Irrational number|Rainbow hats"),
    (3, "3.1", 33, "topic", "Basics of derivatives|Maximum and minimum|L'Hospital's rule"),
    (3, "3.2", 36, "topic", "Basics of integration|Applications of integration|Expected value using integration"),
    (3, "3.3", 40, "topic", "Partial derivatives and multiple integrals"),
    (3, "3.4", 41, "topic", "Taylor series|Newton's method|Lagrange multipliers"),
    (3, "3.5", 46, "topic", "Separable ODE|First-order linear ODE|Homogeneous linear ODE|Nonhomogeneous linear ODE"),
    (3, "3.6", 50, "topic", "Vectors|QR decomposition|Determinant, eigenvalue and eigenvector|Positive semidefinite and positive definite matrices|LU decomposition and Cholesky decomposition"),
    (4, "4.1", 59, "problem", "Coin toss game I|Card game I|Drunk passenger|N points on a circle"),
    (4, "4.2", 64, "problem", "Poker hands|Hopping rabbit|Screwy pirates II|Chess tournament|Application letters|Birthday problem II|100th digit|Cubic integer"),
    (4, "4.3", 72, "problem", "Boys and girls|All-girl world?|Unfair coin|Fair coin from an unfair coin|Dart game|Birthday line|Dice order|Monty Hall|Amoeba population|Candies in a jar|Coin toss game II|Russian roulette|Aces|Gambler's ruin|Basketball scores|Cars on road"),
    (4, "4.4", 86, "problem", "Meeting probability|Probability of triangle|Poisson process property|Normal moments"),
    (4, "4.5", 92, "problem", "Connecting noodles|Optimal hedge ratio|Dice game|Card game II|Sum of random variables|Coupon collection|Joint default"),
    (4, "4.6", 99, "problem", "Expected maximum and minimum|Correlation maximum and minimum|Random ants"),
    (5, "5.1", 105, "problem", "Gambler's ruin II|Dice question|Coin triplets|Color balls"),
    (5, "5.2", 115, "problem", "Drunk man|Dice game II|Ticket line|Coin sequence"),
    (5, "5.3", 121, "topic", "Dynamic programming algorithm"),
    (5, "5.3", 121, "problem", "Dice game III|World series|Dynamic dice game|Dynamic card game"),
    (5, "5.4", 129, "topic", "Brownian motion|Stopping and first passage|Ito's lemma"),
    (6, "6.1", 137, "topic", "Option price directions|Put-call parity|American versus European options|Black-Scholes-Merton PDE|Black-Scholes-Merton formula"),
    (6, "6.2", 149, "topic", "Delta|Gamma|Theta|Vega"),
    (6, "6.3", 158, "topic", "Bull spread|Straddle|Binary option|Exchange option"),
    (6, "6.4", 163, "topic", "Portfolio optimization|Value at Risk|Duration and convexity|Forwards and futures|Interest-rate models"),
    (7, "7.1", 171, "problem", "Number swap|Unique elements|Horner algorithm|Moving average|Sorting|Random permutation|Search|Fibonacci|Maximum contiguous subarray"),
    (7, "7.2", 182, "problem", "Power of two|Multiplication by seven|Probability simulation|Poisonous wine"),
    (7, "7.3", 184, "problem", "Monte Carlo|Finite difference"),
]


BRAINTEASER_HOME_TITLES = {
    # Brainteasers: general reasoning with strong QR interview transfer.
    "Screwy pirates", "Tiger and sheep", "River crossing", "Birthday problem",
    "Card game", "Burning ropes", "Defective ball", "Trailing zeros", "Horse race",
    "Infinite sequence", "Box packing", "Door to offer", "Last ball", "Light switches",
    "Coin piles", "Mislabeled bags", "Wise men", "Missing integers",
    "Calendar cubes", "Message delivery", "Quant salary", "Clock pieces",
    "Counterfeit coins I", "Matching socks", "Handshakes", "Have we met before?",
    "Ants on a square", "Counterfeit coins II", "Prisoner problem", "Division by 9",
    "Chameleon colors", "Coin split problem", "Chocolate bar problem", "Race track",
    "Irrational number", "Rainbow hats",
}

SCORE_GROUPS = {
    (2, 3, 3, 2): BRAINTEASER_HOME_TITLES,
    (3, 2, 3, 3): {
    # Calculus and linear algebra.
    "Maximum and minimum", "Taylor series", "Newton's method", "Lagrange multipliers",
    "QR decomposition", "Determinant, eigenvalue and eigenvector",
    "Positive semidefinite and positive definite matrices", "LU decomposition and Cholesky decomposition",
    },
    (3, 3, 3, 3): {
    # Probability.
    "Drunk passenger", "Poker hands", "Birthday problem II", "Boys and girls",
    "Fair coin from an unfair coin", "Monty Hall", "Gambler's ruin",
    "Meeting probability", "Probability of triangle", "Poisson process property",
    "Normal moments", "Optimal hedge ratio", "Coupon collection", "Joint default",
    "Expected maximum and minimum", "Random ants",
    # Stochastic processes and dynamic programming.
    "Gambler's ruin II", "Coin triplets", "Drunk man", "Ticket line",
    "Dynamic programming algorithm", "Stopping and first passage", "Ito's lemma",
    # Finance.
    "Put-call parity", "American versus European options", "Black-Scholes-Merton PDE",
    "Delta", "Gamma", "Vega", "Portfolio optimization", "Value at Risk",
    "Forwards and futures",
    },
    (2, 2, 3, 2): {
    # Numerical algorithms.
    "Horner algorithm", "Maximum contiguous subarray", "Monte Carlo", "Finite difference",
    },
}
SCORE_OVERRIDES = {
    title: values for values, titles in SCORE_GROUPS.items() for title in titles
}


MAPPINGS = {
    # Chapter 2: all 26 high-priority brainteasers have one home in the new unit.
    title: "lower.brainteasers" for title in BRAINTEASER_HOME_TITLES
}
MAPPINGS.update({
    "Maximum and minimum": "upper.ch05",
    "Taylor series": "upper.ch05",
    "Newton's method": "upper.ch14",
    "Lagrange multipliers": "upper.ch13",
    "QR decomposition": "upper.ch06",
    "Determinant, eigenvalue and eigenvector": "upper.ch06",
    "Positive semidefinite and positive definite matrices": "upper.ch06",
    "LU decomposition and Cholesky decomposition": "upper.ch06",
    "Drunk passenger": "upper.ch07",
    "Coin toss game I": "upper.ch07",
    "Poker hands": "upper.ch07",
    "Birthday problem II": "upper.ch07",
    "Poisson process property": "upper.ch07",
    "Sum of random variables": "upper.ch07",
    "Joint default": "upper.ch07",
    "Fair coin from an unfair coin": "upper.ch08",
    "Unfair coin": "upper.ch08",
    "Monty Hall": "upper.ch08",
    "Russian roulette": "upper.ch08",
    "Aces": "upper.ch08",
    "Cars on road": "upper.ch08",
    "Meeting probability": "upper.ch07",
    "Coupon collection": "upper.ch07",
    "Expected maximum and minimum": "upper.ch09",
    "Normal moments": "upper.ch09",
    "Correlation maximum and minimum": "upper.ch09",
    "Coin triplets": "upper.ch11",
    "Gambler's ruin II": "upper.ch11",
    "Drunk man": "upper.ch11",
    "Ticket line": "upper.ch11",
    "World series": "upper.ch11",
    "Stopping and first passage": "lower.derivatives-stochastic",
    "Put-call parity": "lower.derivatives-numerics",
    "Portfolio optimization": "lower.portfolio-risk-optimization",
    "Value at Risk": "lower.portfolio-risk-tail",
    "Finite difference": "upper.ch14",
    "Monte Carlo": "upper.ch14",
})


def scores(chapter: int, title: str) -> dict[str, int]:
    if title not in SCORE_OVERRIDES:
        return {"qr_relevance": 1 if chapter == 2 else 2, "interview_frequency": 1,
                "transferability": 2, "curriculum_fit": 1}
    values = SCORE_OVERRIDES[title]
    return dict(zip(("qr_relevance", "interview_frequency", "transferability", "curriculum_fit"), values))


def render_payload() -> tuple[dict, str]:
    entries = []
    chapter_counts: dict[int, int] = {}
    for chapter, section, page, kind, raw_titles in GROUPS:
        for title in raw_titles.split("|"):
            chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
            source_id = f"GB-{chapter}-{chapter_counts[chapter]:02d}"
            axes = scores(chapter, title)
            total = sum(axes.values())
            entries.append({
                "id": source_id, "source_chapter": chapter, "source_section": section,
                "toc_page": page, "kind": kind, "original_title": title,
                "scores": axes, "total": total,
                "high_priority": kind == "problem" and total >= 9 and axes["qr_relevance"] > 0,
                "home_unit": MAPPINGS.get(title),
            })

    high = [item for item in entries if item["high_priority"]]
    mapped_high = [item for item in high if item["home_unit"]]
    included = [item for item in entries if item["home_unit"]]
    coverage = len(included) / len(entries)
    payload = {
        "schema_version": 1,
        "source": {
            "title": "A Practical Guide to Quantitative Finance Interviews",
            "author": "Xinfeng Zhou", "edition": "2008",
            "catalog_scope": "all named contents entries; every frozen entry enters the coverage denominator",
        },
        "scoring_rule": {
            "axes": ["qr_relevance", "interview_frequency", "transferability", "curriculum_fit"],
            "scale": "0-3", "high_priority": "kind == problem and total >= 9 and qr_relevance > 0",
            "target_coverage": 0.50, "publication_floor": 0.50,
        },
        "summary": {"catalogued": len(entries), "high_priority": len(high),
                    "mapped_high_priority": len(mapped_high), "mapped_entries": len(included),
                    "coverage": round(coverage, 6)},
        "problems": entries,
    }
    lines = [
        "% Generated by tools/build_interview_problem_bank.py; do not edit.",
        "\\begin{center}",
        "\\begin{tabular}{lr}",
        "具名目录条目 & %d \\\\" % len(entries),
        "已纳入本书的条目 & %d \\\\" % len(included),
        "总题量覆盖率 & %.1f\\%% \\\\" % (coverage * 100),
        "\\end{tabular}", "\\end{center}",
        "本统计以冻结后的 146 个具名目录条目为分母；每条只设一个主归属，交叉引用不重复计数。",
        "\\begin{longtable}{p{0.14\\textwidth}p{0.38\\textwidth}p{0.34\\textwidth}}",
        "编号 & 原书题名 & 本书唯一主归属 \\\\ \\hline",
    ]
    for item in included:
        title = item["original_title"].replace("&", "\\&")
        lines.append(f'{item["id"]} & {title} & \\path{{{item["home_unit"]}}} \\\\')
    lines.extend(["\\end{longtable}", ""])
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail when generated files are stale")
    args = parser.parse_args()
    payload, tex = render_payload()
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    out = ROOT / "curriculum" / "interview-problem-ledger.json"
    generated = ROOT / "tex" / "generated" / "interview-problem-coverage.tex"
    if args.check:
        stale = []
        if not out.is_file() or out.read_text(encoding="utf-8") != json_text:
            stale.append(out.relative_to(ROOT).as_posix())
        if not generated.is_file() or generated.read_text(encoding="utf-8") != tex:
            stale.append(generated.relative_to(ROOT).as_posix())
        if stale:
            print("stale generated files: " + ", ".join(stale), file=sys.stderr)
            return 1
    else:
        out.write_text(json_text, encoding="utf-8")
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(tex, encoding="utf-8")
    coverage = payload["summary"]["coverage"]
    print(
        f"catalogued={len(payload['problems'])} "
        f"mapped={payload['summary']['mapped_entries']} coverage={coverage:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
