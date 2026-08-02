"""Build deterministic vector figures without coupling them to LaTeX builds."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "figures" / "figure-specs.json"
ASSET_DIR = ROOT / "tex" / "figures" / "generated"
MANIFEST_PATH = ROOT / "figures" / "figure-manifest.json"
PDF_TIMESTAMP = datetime(2026, 8, 1, tzinfo=timezone.utc)

BLUE = "#2B6CB0"
GREEN = "#2F855A"
ORANGE = "#C05621"
GRAY = "#718096"
LIGHT_BLUE = "#EBF8FF"
LIGHT_GREEN = "#F0FFF4"
LIGHT_ORANGE = "#FFFAF0"
LIGHT_GRAY = "#F7FAFC"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#2D3748",
            "axes.edgecolor": GRAY,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "xtick.color": "#4A5568",
            "ytick.color": "#4A5568",
            "grid.color": "#CBD5E0",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def role_colors(role: str) -> tuple[str, str]:
    return {
        "object": (LIGHT_BLUE, BLUE),
        "valid": (LIGHT_GREEN, GREEN),
        "risk": (LIGHT_ORANGE, ORANGE),
        "background": (LIGHT_GRAY, GRAY),
    }[role]


def clean_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True)


def render_flow(spec: dict) -> plt.Figure:
    nodes = spec["nodes"]
    roles = spec.get("roles", ["object"] * len(nodes))
    fig, ax = plt.subplots(figsize=(7.2, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(nodes))
    width = min(0.17, 0.72 / len(nodes))
    for i, (x, label, role) in enumerate(zip(xs, nodes, roles, strict=True)):
        face, edge = role_colors(role)
        border_style = "--" if role == "risk" else (":" if role == "background" else "-")
        border_width = 2.2 if role == "valid" else 1.6
        box = FancyBboxPatch(
            (x - width / 2, 0.38), width, 0.24,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=face, edgecolor=edge, linewidth=border_width,
            linestyle=border_style,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=8.3, wrap=True)
        if i:
            prev_role = roles[i - 1]
            color = ORANGE if role == "risk" else (GREEN if role == "valid" else BLUE)
            style = "--" if role == "risk" or prev_role == "risk" else "-"
            ax.add_patch(
                FancyArrowPatch(
                    (xs[i - 1] + width / 2 + 0.008, 0.5),
                    (x - width / 2 - 0.008, 0.5),
                    arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
                    linestyle=style, color=color,
                )
            )
    return fig


def render_architecture(spec: dict) -> plt.Figure:
    proxy = {"nodes": spec["layers"], "roles": spec.get("roles", [])}
    return render_flow(proxy)


def render_graph() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.set_aspect("equal")
    ax.axis("off")
    positions = np.array([[0.08, 0.50], [0.28, 0.80], [0.31, 0.24], [0.55, 0.62], [0.72, 0.24], [0.90, 0.58]])
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
    for a, b in edges:
        ax.plot(*zip(positions[a], positions[b]), color=GRAY, linewidth=1.4, zorder=1)
    for i, (x, y) in enumerate(positions):
        role = "valid" if i == 3 else "object"
        face, edge = role_colors(role)
        ax.add_patch(Circle((x, y), 0.055, facecolor=face, edgecolor=edge, linewidth=1.8, zorder=2))
        ax.text(x, y, f"v{i}", ha="center", va="center", zorder=3)
    ax.annotate("aggregate neighbor messages", (0.55, 0.73), ha="center", color=GREEN, fontsize=9)
    ax.annotate("edge meaning + timestamp", (0.55, 0.05), ha="center", color=ORANGE, fontsize=8.5)
    return fig


def render_timeline(spec: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 1.8))
    ax.axis("off")
    total = sum(float(segment[1]) for segment in spec["segments"])
    cursor = 0.0
    for label, width, role in spec["segments"]:
        width = float(width)
        face, edge = role_colors(role)
        hatch = "///" if role == "risk" else (".." if role == "valid" else None)
        ax.barh(
            0, width, left=cursor, height=0.42, color=face,
            edgecolor=edge, linewidth=1.5, hatch=hatch,
        )
        ax.text(cursor + width / 2, 0, label, ha="center", va="center", fontsize=8.5, color=edge)
        cursor += width
    ax.annotate("time", (total, -0.36), ha="right", color=GRAY)
    ax.set(xlim=(0, total), ylim=(-0.5, 0.5))
    return fig


def render_evidence(scene: str) -> plt.Figure:
    rng = np.random.default_rng(20260801)

    if scene == "projection":
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        origin = np.array([0.0, 0.0]); observed = np.array([2.2, 1.65]); projection = np.array([2.2, 0.0])
        ax.axhline(0, color=GRAY, linewidth=1.3, label="column space")
        ax.arrow(*origin, *observed, color=BLUE, width=.018, length_includes_head=True)
        ax.arrow(*origin, *projection, color=GREEN, width=.018, length_includes_head=True)
        ax.arrow(*projection, *(observed-projection), color=ORANGE, linestyle="--", width=.012, length_includes_head=True)
        ax.text(2.23, 1.7, "y", color=BLUE); ax.text(1.35, -.18, "projection", color=GREEN); ax.text(2.28, .82, "residual", color=ORANGE)
        ax.plot([2.05,2.05,2.2],[0,.15,.15],color=GRAY)
        ax.axis("off"); ax.set(xlim=(-.2,3),ylim=(-.35,2.1)); ax.set_aspect("equal")
        return fig

    if scene == "convergence-map":
        fig, ax = plt.subplots(figsize=(6.6, 3.2)); ax.axis("off")
        positions = {"a.s.":(.12,.72), "L^p":(.12,.25), "probability":(.52,.52), "distribution":(.88,.52)}
        for label,(x,y) in positions.items():
            role="valid" if label in {"probability","distribution"} else "object"; face,edge=role_colors(role)
            ax.add_patch(FancyBboxPatch((x-.09,y-.08),.18,.16,boxstyle="round,pad=.02",facecolor=face,edgecolor=edge,linewidth=1.6)); ax.text(x,y,label,ha="center",va="center")
        for start,end in [("a.s.","probability"),("L^p","probability"),("probability","distribution")]:
            ax.add_patch(FancyArrowPatch(positions[start],positions[end],arrowstyle="-|>",mutation_scale=12,color=GREEN,linewidth=1.7))
        ax.add_patch(FancyArrowPatch(positions["probability"],positions["a.s."],connectionstyle="arc3,rad=.35",arrowstyle="-|>",mutation_scale=12,color=ORANGE,linewidth=1.5,linestyle="--"))
        ax.text(.36,.92,"subsequence only",color=ORANGE,ha="center",fontsize=8.5)
        ax.set(xlim=(0,1),ylim=(0,1))
        return fig

    if scene == "simple-integral":
        fig, ax = plt.subplots(figsize=(6.6, 3.1))
        x = np.linspace(0, 3, 500)
        y = np.exp(-0.7 * x) * (1 + 0.18 * np.sin(5 * x))
        ax.plot(x, y, color=BLUE, linewidth=2, label=r"$f(x)$")
        for n, alpha in [(6, 0.18), (12, 0.30)]:
            bins = np.linspace(0, 3, n + 1)
            heights = [y[(x >= bins[i]) & (x < bins[i + 1])].min() for i in range(n)]
            ax.stairs(heights, bins, baseline=0, fill=True, alpha=alpha, color=GREEN, label=f"simple n={n}")
        clean_axes(ax); ax.set(xlabel="x", ylabel="height", ylim=(0, 1.15)); ax.legend(ncol=3)
        return fig

    if scene == "lp-convergence":
        fig, ax = plt.subplots(figsize=(6.6, 3.1))
        x = np.linspace(0, 1, 800)
        for n, style in [(4, "-"), (10, "--"), (30, ":")]:
            y = np.sqrt(n) * np.exp(-n**2 * (x - 0.5) ** 2)
            ax.plot(x, y, linestyle=style, linewidth=1.8, label=f"n={n}")
        clean_axes(ax); ax.set(xlabel="x", ylabel=r"$f_n(x)$"); ax.legend()
        return fig

    if scene == "hessian":
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        x = np.linspace(-2.5, 2.5, 180); y = np.linspace(-2, 2, 180)
        X, Y = np.meshgrid(x, y); Z = 0.5 * (4 * X**2 + Y**2 + 1.2 * X * Y)
        ax.contour(X, Y, Z, levels=9, colors=BLUE, linewidths=1)
        ax.arrow(1.4, 0.8, -0.85, -0.75, width=0.018, color=GREEN, length_includes_head=True)
        ax.text(1.35, 1.05, "-gradient", color=GREEN)
        hessian = np.array([[4.0, 0.6], [0.6, 1.0]])
        _, directions = np.linalg.eigh(hessian)
        high_curvature = directions[:, -1]
        ax.axline(
            (0, 0), slope=high_curvature[1] / high_curvature[0],
            color=ORANGE, linestyle="--", label="high curvature",
        )
        ax.set_aspect("equal"); clean_axes(ax); ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$"); ax.legend()
        return fig

    if scene == "conditioning":
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        x = np.linspace(-2.5, 2.5, 90); noise = rng.normal(0, 0.65, x.size); y = 0.7 * x + noise
        ax.scatter(x, y, s=12, alpha=0.55, color=GRAY, label="observations")
        ax.plot(x, 0.7 * x, color=GREEN, linewidth=2.3, label=r"$E[Y\mid X]$")
        for i in [15, 35, 58, 75]:
            ax.plot([x[i], x[i]], [0.7*x[i], y[i]], color=ORANGE, linestyle="--", linewidth=1)
        clean_axes(ax); ax.set(xlabel="visible information X", ylabel="Y"); ax.legend()
        return fig

    if scene == "multiple-testing":
        fig, ax = plt.subplots(figsize=(6.5, 3.1))
        m = np.array([1, 5, 10, 25, 50, 100, 250, 500])
        probability = 1 - (1 - 0.05) ** m
        ax.plot(m, probability, marker="o", color=ORANGE, linewidth=2, label=r"$P(\min p < .05)$ under null")
        ax.plot(m, np.full_like(m, 0.05, dtype=float), linestyle="--", color=GREEN, label="family target 5%")
        ax.set_xscale("log"); clean_axes(ax); ax.set(xlabel="number of tests", ylabel="probability", ylim=(0, 1.03)); ax.legend()
        return fig

    if scene == "kkt":
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        x = np.linspace(-0.2, 2.5, 200); y = np.linspace(-0.2, 2.2, 200)
        X, Y = np.meshgrid(x, y); Z = (X - 1.8)**2 + 1.4*(Y - 1.4)**2
        ax.contour(X, Y, Z, levels=[0.2, 0.6, 1.2, 2.2, 3.5], colors=BLUE)
        ax.fill_between(x, 0, 1.25 - 0.55*x, where=(1.25-0.55*x)>=0, color=LIGHT_GREEN, alpha=0.9)
        ax.plot(x, 1.25 - 0.55*x, color=GREEN, linewidth=2, label="active constraint")
        # Minimize the displayed quadratic on y = 1.25 - 0.55 x.
        x_star = (3.6 + 2.8 * 0.55 * (1.25 - 1.4)) / (2 + 2.8 * 0.55**2)
        point = np.array([x_star, 1.25 - 0.55 * x_star])
        gradient = np.array([2 * (point[0] - 1.8), 2.8 * (point[1] - 1.4)])
        arrow = 0.62 * gradient / np.linalg.norm(gradient)
        ax.scatter(*point, color=ORANGE, s=45, zorder=5, label="KKT point")
        ax.arrow(*point, *arrow, color=ORANGE, width=0.012, length_includes_head=True)
        clean_axes(ax); ax.set(xlim=(-.1, 2.3), ylim=(-.1, 2.0), xlabel=r"$w_1$", ylabel=r"$w_2$"); ax.legend()
        return fig

    if scene == "cancellation":
        fig, ax = plt.subplots(figsize=(6.4, 3.1))
        eps = np.logspace(-2, -15, 100)
        direct = np.abs(((1 + eps) - 1) / eps - 1)
        stable = np.full_like(eps, np.finfo(float).eps)
        ax.loglog(eps, np.maximum(direct, 1e-18), color=ORANGE, linewidth=2, label="subtractive form")
        ax.loglog(eps, stable, color=GREEN, linestyle="--", linewidth=2, label="stable reformulation")
        clean_axes(ax); ax.set(xlabel=r"separation $\epsilon$", ylabel="relative error"); ax.legend()
        return fig

    if scene == "mc-rate":
        fig, ax = plt.subplots(figsize=(6.2, 3.1))
        n = np.logspace(2, 6, 9); se = 0.8 / np.sqrt(n)
        ax.loglog(n, se, marker="o", color=BLUE, label=r"standard error $\propto N^{-1/2}$")
        ax.loglog(n, 0.8/n, linestyle="--", color=GRAY, label=r"reference $N^{-1}$")
        clean_axes(ax); ax.set(xlabel="simulation count N", ylabel="error scale"); ax.legend()
        return fig

    if scene == "audit-matrix":
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        matrix = np.array([[1,1,1,0,1],[1,1,0,1,1],[1,0,1,1,0],[1,1,1,1,1]], dtype=float)
        cmap = mpl.colors.ListedColormap([LIGHT_ORANGE, LIGHT_GREEN])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(5), ["data", "math", "numerics", "friction", "reproduce"])
        ax.set_yticks(range(4), ["claim A", "claim B", "claim C", "release"])
        for i in range(4):
            for j in range(5):
                ax.text(j, i, "PASS" if matrix[i,j] else "GAP", ha="center", va="center", color=GREEN if matrix[i,j] else ORANGE, weight="bold")
        ax.tick_params(length=0)
        return fig

    if scene == "ic-decay":
        fig, ax = plt.subplots(figsize=(6.2, 3.1))
        h = np.arange(1, 21); ic = 0.08*np.exp(-h/6); cost = 0.011 + 0.0018*h
        ax.plot(h, ic, marker="o", color=BLUE, label="signal IC")
        ax.plot(h, cost/100, color=ORANGE, linestyle="--", label="cost scale (normalized)")
        clean_axes(ax); ax.set(xlabel="forecast horizon", ylabel="scale"); ax.legend()
        return fig

    if scene == "cost-chain":
        fig, ax = plt.subplots(figsize=(6.4, 3.1))
        stages = ["gross", "neutralize", "turnover", "fees", "impact", "net"]
        values = [100, 92, 81, 75, 63, 63]
        colors = [BLUE, BLUE, ORANGE, ORANGE, ORANGE, GREEN]
        ax.bar(stages, values, color=colors, alpha=0.85)
        ax.plot(stages, values, color=GRAY, linewidth=1)
        ax.set_ylim(0, 108); clean_axes(ax); ax.set(ylabel="return index")
        return fig

    if scene == "cointegration":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True, gridspec_kw={"height_ratios":[2,1]})
        t = np.arange(180); trend = np.cumsum(rng.normal(0, 0.35, t.size)); spread = np.zeros_like(trend)
        for i in range(1, t.size): spread[i] = 0.86*spread[i-1] + rng.normal(0, 0.18)
        p1 = 100 + trend + spread; p2 = 99 + trend
        ax1.plot(t, p1, color=BLUE, label="price A"); ax1.plot(t, p2, color=GRAY, linestyle="--", label="price B")
        ax2.plot(t, p1-p2-1, color=GREEN, label="spread"); ax2.axhline(0, color=GRAY, linewidth=0.8)
        clean_axes(ax1); clean_axes(ax2); ax1.legend(); ax2.legend(); ax2.set_xlabel("time")
        return fig

    if scene == "ou":
        fig, ax = plt.subplots(figsize=(6.4, 3.1))
        t = np.linspace(0, 20, 300)
        for x0, style in [(2.2, "-"), (-1.8, "--")]:
            mean = x0*np.exp(-0.23*t)
            ax.plot(t, mean, linestyle=style, color=BLUE, linewidth=2)
        half = np.log(2)/0.23; ax.axvline(half, color=GREEN, linestyle=":", linewidth=2, label="half-life")
        ax.axhline(0, color=GRAY, linewidth=1); clean_axes(ax); ax.set(xlabel="time", ylabel=r"$E[X_t\mid X_0]$"); ax.legend()
        return fig

    if scene == "position-pnl":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.0), sharex=True)
        t = np.arange(60); score = np.sin(t/6) + 0.35*np.sin(t/2.5); position = np.clip(score, -0.8, 0.8)
        ret = 0.003*np.roll(score, 1) + rng.normal(0, 0.004, t.size); turnover = np.abs(np.diff(position, prepend=0)); net = position*np.roll(ret,-1)-0.0007*turnover
        ax1.plot(t, score, color=BLUE, label="forecast"); ax1.step(t, position, color=GREEN, where="mid", label="position")
        ax2.plot(t, np.cumsum(net), color=GREEN, label="cumulative net"); ax2.fill_between(t, np.cumsum(net), color=LIGHT_GREEN)
        clean_axes(ax1); clean_axes(ax2); ax1.legend(ncol=2); ax2.legend(); ax2.set_xlabel("decision time")
        return fig

    if scene == "tree-split":
        fig, ax = plt.subplots(figsize=(6.2, 3.1))
        x = np.linspace(-2.5, 2.5, 80); y = (x > 0.35).astype(float) + rng.normal(0, 0.18, x.size)
        ax.scatter(x[x<=.35], y[x<=.35], s=16, color=BLUE, alpha=.7, label="left node")
        ax.scatter(x[x>.35], y[x>.35], s=16, color=GREEN, alpha=.7, label="right node")
        ax.axvspan(.27, .43, color=LIGHT_GREEN); ax.axvline(.35, color=GREEN, linewidth=2, label="chosen threshold")
        clean_axes(ax); ax.set(xlabel="feature value", ylabel="target"); ax.legend(ncol=3)
        return fig

    if scene == "training-curves":
        fig, ax = plt.subplots(figsize=(6.2, 3.1))
        e = np.arange(1, 81); train = .75*np.exp(-e/28)+.08; valid = .62*np.exp(-e/22)+.14 + .00011*(e-43).clip(0)**2
        ax.plot(e, train, color=BLUE, linestyle="-", label="train")
        ax.plot(e, valid, color=ORANGE, linestyle="--", label="validation")
        best = np.argmin(valid)+1; ax.axvline(best, color=GREEN, linestyle="--", label=f"early stop {best}")
        clean_axes(ax); ax.set(xlabel="epoch", ylabel="loss"); ax.legend()
        return fig

    if scene == "attention-mask":
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        n = 8; weights = np.tril(np.fromfunction(lambda i,j: np.exp(-0.45*(i-j)), (n,n)))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        im = ax.imshow(weights, cmap=mpl.colors.LinearSegmentedColormap.from_list("mfq", [LIGHT_GRAY, BLUE]), vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if j > i: ax.text(j, i, "x", ha="center", va="center", color=ORANGE, fontsize=8)
        ax.set(xlabel="key position", ylabel="query position"); fig.colorbar(im, ax=ax, fraction=.046)
        return fig

    if scene == "calibration":
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        p = np.linspace(.05,.95,10); calibrated = p + np.array([.01,-.02,.01,.02,-.01,.01,-.02,.01,.02,-.01]); over = np.sqrt(p)*.88
        ax.plot([0,1],[0,1], color=GRAY, linestyle="--", label="ideal")
        ax.plot(p, calibrated, marker="o", color=GREEN, label="calibrated")
        ax.plot(p, over, marker="s", color=ORANGE, label="over-confident")
        clean_axes(ax); ax.set(xlim=(0,1), ylim=(0,1), xlabel="predicted probability", ylabel="observed frequency"); ax.legend()
        return fig

    if scene == "drift":
        fig, ax = plt.subplots(figsize=(6.4, 3.1))
        t = np.arange(100); input_shift = 0.15+0.006*t+0.05*np.sin(t/8); performance = .68-.0015*t-.12*(t>62)
        ax.plot(t, input_shift, color=BLUE, linestyle="-", label="input drift index")
        ax.plot(t, performance, color=ORANGE, linestyle="--", label="out-of-sample score")
        ax.axvline(62, color=GREEN, linestyle="--", label="response trigger")
        clean_axes(ax); ax.set(xlabel="calendar time", ylabel="monitor value"); ax.legend()
        return fig

    if scene == "ml-decision-pnl":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.1), sharex=True)
        t = np.arange(70); score = .9*np.sin(t/7)+rng.normal(0,.18,t.size); position=np.where(np.abs(score)>.35,np.sign(score),0)
        future=.003*np.roll(score,1)+rng.normal(0,.006,t.size); cost=.0006*np.abs(np.diff(position,prepend=0)); net=position*np.roll(future,-1)-cost
        ax1.plot(t,score,color=BLUE,label="model score"); ax1.step(t,position,color=GREEN,where="mid",label="position"); ax1.axhspan(-.35,.35,color=LIGHT_GRAY)
        ax2.plot(t,np.cumsum(net),color=GREEN,label="net PnL"); ax2.plot(t,np.cumsum(position*np.roll(future,-1)),color=GRAY,linestyle="--",label="gross PnL")
        clean_axes(ax1); clean_axes(ax2); ax1.legend(ncol=2); ax2.legend(ncol=2); ax2.set_xlabel("decision time")
        return fig

    if scene == "quadratic-variation":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
        n=4096; dt=1/n; path=np.r_[0,np.cumsum(rng.normal(0,np.sqrt(dt),n))]; t=np.linspace(0,1,n+1)
        ax1.plot(t,path,color=BLUE,linewidth=1); ax1.set(xlabel="t",ylabel=r"$W_t$",title="one path")
        ns=np.array([8,16,32,64,128,256,512,1024]); qv=[]
        for m in ns:
            idx=np.linspace(0,n,m+1,dtype=int); qv.append(np.sum(np.diff(path[idx])**2))
        ax2.plot(ns,qv,marker="o",color=GREEN,label="quadratic sum"); ax2.axhline(1,color=GRAY,linestyle="--",label="T=1")
        ax2.set_xscale("log",base=2); ax2.set(xlabel="partition count",ylabel="sum",title="nested partitions"); ax2.legend()
        clean_axes(ax1); clean_axes(ax2)
        return fig

    if scene == "vol-surface":
        fig = plt.figure(figsize=(6.4, 4.2)); ax = fig.add_subplot(111, projection="3d")
        k=np.linspace(.75,1.25,40); tau=np.linspace(.05,2,35); K,T=np.meshgrid(k,tau)
        vol=.18+.16*(K-1)**2+.07*np.exp(-T)-.035*(K-1)
        ax.plot_surface(K,T,vol,cmap=mpl.colors.LinearSegmentedColormap.from_list("mfq",[LIGHT_GREEN,BLUE]),linewidth=0,antialiased=True,alpha=.95)
        ax.set(xlabel="moneyness K/F",ylabel="maturity",zlabel="implied vol"); ax.view_init(27,-55)
        return fig

    if scene == "pde-grid":
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        s=np.linspace(0,2,80); tau=np.linspace(0,1,55); S,T=np.meshgrid(s,tau); value=np.maximum(S-1,0)*np.exp(-.04*T)+.18*np.sqrt(T)*np.exp(-4*(S-1)**2)
        im=ax.imshow(value,origin="lower",aspect="auto",extent=[0,2,0,1],cmap=mpl.colors.LinearSegmentedColormap.from_list("mfq",[LIGHT_GRAY,LIGHT_BLUE,BLUE]))
        ax.plot([1,1],[0,1],color=ORANGE,linestyle="--",linewidth=1,label="strike"); ax.set(xlabel="spot grid",ylabel="time to maturity"); ax.legend(); fig.colorbar(im,ax=ax,label="option value")
        return fig

    if scene == "greeks":
        fig, axes = plt.subplots(1,3,figsize=(7.2,2.6),sharex=True)
        s=np.linspace(.55,1.45,200); z=(s-1)/.16; delta=1/(1+np.exp(-7*(s-1))); gamma=np.exp(-.5*z*z)/(.16*np.sqrt(2*np.pi)); vega=.28*np.exp(-.5*z*z)
        for ax,y,title,color in zip(axes,[delta,gamma,vega],["Delta","Gamma","Vega"],[BLUE,ORANGE,GREEN],strict=True):
            ax.plot(s,y,color=color,linewidth=2); ax.axvline(1,color=GRAY,linestyle="--"); ax.set_title(title); clean_axes(ax); ax.set_xlabel("S/K")
        return fig

    if scene == "hedge-error":
        fig, ax = plt.subplots(figsize=(6.3,3.1))
        daily=rng.normal(-.001,.038,4000); weekly=rng.normal(-.003,.072,4000)
        bins=np.linspace(-.22,.22,55)
        ax.hist(weekly,bins=bins,density=True,alpha=.32,color=ORANGE,hatch="///",label="coarse hedge")
        ax.hist(daily,bins=bins,density=True,alpha=.48,color=GREEN,histtype="step",linewidth=1.6,label="frequent hedge")
        ax.axvline(0,color=GRAY,linewidth=1); clean_axes(ax); ax.set(xlabel="terminal hedge error",ylabel="density"); ax.legend()
        return fig

    if scene == "risk-ellipse":
        fig, ax = plt.subplots(figsize=(5.6,3.7)); x=np.linspace(-2,2,180); y=np.linspace(-2,2,180); X,Y=np.meshgrid(x,y); Z=1.8*X**2+.65*Y**2+1.15*X*Y
        risk_matrix = np.array([[1.8, 0.575], [0.575, 0.65]])
        _, directions = np.linalg.eigh(risk_matrix)
        dominant = directions[:, -1]
        if dominant[0] < 0:
            dominant = -dominant
        arrow = 1.28 * dominant
        ax.contour(X,Y,Z,levels=[.5,1,2,3,4],colors=[GREEN,BLUE,BLUE,GRAY,GRAY]); ax.arrow(0,0,*arrow,color=ORANGE,width=.018,length_includes_head=True); ax.text(0.76*arrow[0],0.76*arrow[1]+.12,"dominant risk",color=ORANGE)
        ax.set_aspect("equal"); clean_axes(ax); ax.set(xlabel=r"weight $w_1$",ylabel=r"weight $w_2$")
        return fig

    if scene == "shrinkage":
        fig, ax = plt.subplots(figsize=(6.2,3.1)); lam=np.linspace(0,1,100); variance=.9*(1-lam)**2+.08; bias=.42*lam**2; total=variance+bias
        ax.plot(lam,variance,color=BLUE,label="variance"); ax.plot(lam,bias,color=ORANGE,linestyle="--",label="squared bias"); ax.plot(lam,total,color=GREEN,linewidth=2,label="total error")
        best=lam[np.argmin(total)]; ax.axvline(best,color=GREEN,linestyle=":"); clean_axes(ax); ax.set(xlabel="shrinkage intensity",ylabel="error scale"); ax.legend()
        return fig

    if scene == "frontier":
        fig, ax = plt.subplots(figsize=(5.7,3.7)); risk=np.linspace(.06,.28,120); ideal=.055+1.05*np.sqrt(np.maximum(risk-.05,0)); constrained=ideal-.018-.08*(risk-.15)**2; implemented=constrained-.012
        ax.plot(risk,ideal,color=BLUE,label="frictionless"); ax.plot(risk,constrained,color=GREEN,linestyle="--",label="constraints"); ax.plot(risk,implemented,color=ORANGE,linestyle=":",label="after costs")
        clean_axes(ax); ax.set(xlabel="risk",ylabel="expected return"); ax.legend()
        return fig

    if scene == "tail-risk":
        fig, ax = plt.subplots(figsize=(6.2,3.1)); loss=np.r_[rng.normal(0,.9,3600),rng.normal(3.2,1.1,400)]; var=np.quantile(loss,.95); es=loss[loss>=var].mean()
        ax.hist(loss,bins=55,density=True,color=LIGHT_BLUE,edgecolor=BLUE,linewidth=.4); ax.axvline(var,color=ORANGE,linewidth=2,label=f"VaR 95%={var:.2f}"); ax.axvspan(var,loss.max(),color=LIGHT_ORANGE,alpha=.8,label=f"ES={es:.2f}")
        clean_axes(ax); ax.set(xlabel="loss",ylabel="density"); ax.legend()
        return fig

    if scene == "hawkes":
        fig, ax = plt.subplots(figsize=(6.4,3.1)); t=np.linspace(0,20,900); events=np.array([1.2,1.6,2.0,7.5,8.0,8.2,8.5,15.0]); intensity=np.full_like(t,.18)
        for event in events: intensity += .72*np.exp(-1.15*np.maximum(t-event,0))*(t>=event)
        ax.plot(t,intensity,color=BLUE,linewidth=2); ax.vlines(events,0,.16,color=ORANGE,linewidth=1.5,label="events"); ax.axhline(.18,color=GRAY,linestyle="--",label="baseline")
        clean_axes(ax); ax.set(xlabel="event time",ylabel="conditional intensity"); ax.legend()
        return fig

    if scene == "execution":
        fig, ax = plt.subplots(figsize=(6.4,3.1)); t=np.linspace(0,1,80); uniform=1-t; front=(1-t)**1.7; back=1-t**1.7
        ax.plot(t,uniform,color=BLUE,label="uniform"); ax.plot(t,front,color=GREEN,label="front-loaded"); ax.plot(t,back,color=ORANGE,linestyle="--",label="back-loaded")
        ax.fill_between(t,0,front,color=LIGHT_GREEN,alpha=.45); clean_axes(ax); ax.set(xlabel="fraction of horizon",ylabel="remaining inventory"); ax.legend()
        return fig

    raise ValueError(f"Unknown evidence scene: {scene}")


def render(spec: dict) -> plt.Figure:
    if spec["kind"] == "flow":
        return render_flow(spec)
    if spec["kind"] == "architecture":
        return render_architecture(spec)
    if spec["kind"] == "graph":
        return render_graph()
    if spec["kind"] == "timeline":
        return render_timeline(spec)
    if spec["kind"] == "evidence":
        return render_evidence(spec["scene"])
    raise ValueError(f"Unknown figure kind: {spec['kind']}")


def tex_wrapper(spec: dict) -> str:
    asset = f"tex/figures/generated/{spec['id']}.pdf"
    return (
        "% Generated by tools/build_figures.py; edit figures/figure-specs.json instead.\n"
        f"\\MFQFigure{{{asset}}}{{{spec['caption']}}}{{{spec['id']}}}\n"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def companion_for(spec: dict) -> str | None:
    if spec["kind"] != "evidence":
        return None
    if spec["volume"] == "upper":
        matches = sorted((ROOT / "notebooks" / "upper").glob(f"{spec['chapter']}_*.ipynb"))
    else:
        exact = ROOT / "notebooks" / "lower" / f"{spec['chapter'].replace('-', '_')}.ipynb"
        route_fallbacks = {
            "ml-alpha-deep": "ch03_ml_alpha.ipynb",
            "ml-alpha-sequence": "ch03_ml_alpha.ipynb",
            "ml-alpha-frontiers": "ch03_ml_alpha.ipynb",
        }
        matches = [
            exact if exact.is_file()
            else ROOT / "notebooks" / "lower" / route_fallbacks.get(spec["chapter"], "")
        ]
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"No unique companion notebook for evidence figure {spec['id']}")
    return matches[0].relative_to(ROOT).as_posix()


def main() -> int:
    configure_style()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    script_hash = sha256(Path(__file__))
    records = []
    for spec in data["figures"]:
        fig = render(spec)
        pdf_path = ASSET_DIR / f"{spec['id']}.pdf"
        fig.savefig(
            pdf_path,
            format="pdf",
            metadata={
                "Title": spec["id"],
                "Author": "math-for-quant",
                "Creator": "tools/build_figures.py",
                "CreationDate": PDF_TIMESTAMP,
                "ModDate": PDF_TIMESTAMP,
            },
        )
        plt.close(fig)
        tex_path = ASSET_DIR / f"{spec['id']}.tex"
        tex_path.write_text(tex_wrapper(spec), encoding="utf-8", newline="\n")
        records.append(
            {
                "id": spec["id"],
                "volume": spec["volume"],
                "chapter": spec["chapter"],
                "kind": spec["kind"],
                "scene": spec.get("scene"),
                "source": "tools/build_figures.py",
                "spec": "figures/figure-specs.json",
                "companion": companion_for(spec),
                "cached_asset": pdf_path.relative_to(ROOT).as_posix(),
                "tex_wrapper": tex_path.relative_to(ROOT).as_posix(),
                "caption": spec["caption"],
                "source_sha256": script_hash,
                "asset_sha256": sha256(pdf_path),
            }
        )
    manifest = {"schema_version": 1, "figure_count": len(records), "figures": records}
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    generated_count = len(data["figures"])
    print(f"built {generated_count} cached vector figures; registered {len(records)} total figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
