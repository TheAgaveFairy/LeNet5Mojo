#!/usr/bin/env python3
"""Generate the poster's charts as SVG.

Every number here traces to a file in one of the two repos; the SOURCE comment
above each dataset says which. Run from poster/:  python figures/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE
FONTS = HERE.parent / "fonts"

# Register Atkinson Hyperlegible so charts match the poster body type.
for f in FONTS.glob("*.otf"):
    font_manager.fontManager.addfont(str(f))

SANS = "Atkinson Hyperlegible Next"
MONO = "Atkinson Hyperlegible Mono"

# ---------------------------------------------------------------- palette ----
INK = "#1C1B1A"
MUTED = "#6B6862"
RULE = "#CFC9C0"
PAPER = "#FAF8F5"

C = {
    "mojo":   "#D9541E",   # Mojo, 12 streams — the accent colour
    "mojo1":  "#E9A473",   # Mojo, single stream
    "trt":    "#3E6D9C",
    "jax":    "#7B5EA7",
    "torch":  "#B44B5C",
    "ort":    "#8A8A82",
    "max":    "#2E8B8B",
}

plt.rcParams.update({
    "font.family": SANS,
    "font.size": 15,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.4,
    "lines.linewidth": 3.2,
    "svg.fonttype": "none",     # keep text as text — stays vector + searchable
})


def millions(x, _pos):
    if x == 0:
        return "0"
    return f"{x/1e6:.1f}M"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", p.relative_to(HERE.parent))


# ============================================================== FIGURE 1 ====
# SOURCE: CNNTesting/results/results.csv (clocks locked, streaming scenario).
# x = effective batch = batch_size * num_streams for the Mojo rows.
F1 = {
    "Mojo — 12 streams": (
        [120, 300, 600, 1200, 1500, 2400],
        [816794, 1258860, 1370354, 1414895, 1409675, 1458512],
        C["mojo"], "o", 3.6,
    ),
    "Mojo — 1 stream": (
        [1, 8, 50, 200, 500, 1000, 2000],
        [19764, 141105, 598998, 1030211, 1206012, 1261033, 1334828],
        C["mojo1"], "o", 3.0,
    ),
    "ONNX-RT + TensorRT": (
        [1, 8, 32, 128, 512, 1024, 2048],
        [16407, 95170, 321601, 710587, 1103067, 1286873, 1415433],
        C["trt"], "s", 3.0,
    ),
    "JAX": (
        [1, 8, 32, 128, 512, 1024, 2048],
        [3054, 24374, 97817, 354698, 894513, 1071687, 1223198],
        C["jax"], "^", 2.4,
    ),
    "PyTorch (CUDA)": (
        [1, 8, 32, 128, 512, 1024, 2048],
        [5934, 46157, 180379, 435661, 675541, 764531, 799863],
        C["torch"], "v", 2.4,
    ),
    "ONNX Runtime": (
        [1, 8, 32, 128, 512, 1024, 2048],
        [7938, 61989, 206906, 435132, 647170, 752727, 783990],
        C["ort"], "D", 2.4,
    ),
}


def fig1():
    fig, ax = plt.subplots(figsize=(14.5, 4.3))

    # The regime where the Mojo lead is outside session-to-session variance.
    ax.axvspan(0.8, 512, color="#D9541E", alpha=0.055, zorder=0, lw=0)
    ax.axvline(512, color=C["mojo"], lw=1.6, ls=(0, (5, 4)), alpha=0.55, zorder=1)

    for label, (xs, ys, colour, marker, lw) in F1.items():
        z = 5 if "Mojo" in label else 3
        ax.plot(xs, ys, color=colour, marker=marker, markersize=9 if z == 5 else 7,
                lw=lw, label=label, zorder=z, markeredgecolor=PAPER,
                markeredgewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlim(0.8, 3000)
    ax.set_ylim(0, 1.58e6)
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_xticks([1, 8, 32, 128, 512, 2048])
    ax.set_xticklabels(["1", "8", "32", "128", "512", "2048"])
    ax.set_xlabel("effective batch size  (images in flight)", labelpad=10)
    ax.set_ylabel("throughput  (images / s)", labelpad=10)
    ax.grid(axis="y", color=RULE, lw=1.0, alpha=0.7)
    ax.set_axisbelow(True)

    # Annotations and legend are placed in the two regions no curve reaches:
    # upper-left (x < ~4) and lower-right (x > ~600, y < 0.55M).
    ax.annotate("Mojo leads across this whole regime\n— 1.48× at batch 8",
                xy=(8, 141105), xytext=(1.05, 3.1e5),
                fontsize=14.5, color=C["mojo"], weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C["mojo"], lw=2.0,
                                shrinkA=4, shrinkB=6))
    ax.annotate("statistical tie at peak\n(+3.0%, inside the ±10%\nsession wobble)",
                xy=(2300, 1.42e6), xytext=(640, 2.4e5),
                fontsize=13.5, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                                shrinkA=4, shrinkB=6))

    leg = ax.legend(loc="upper left", frameon=True, fontsize=13.5,
                    borderpad=0.55, labelspacing=0.30, handlelength=2.0)
    leg.get_frame().set_facecolor(PAPER)
    leg.get_frame().set_edgecolor(RULE)
    save(fig, "fig1_throughput_vs_batch.svg")


# ============================================================== FIGURE 2 ====
# SOURCE: CNNTesting/results/results.csv, batch_size=1 rows. us/img = 1e6/fps.
# MAX is shown as its BEST configuration (--filter-layout fcrs --pool-reshape,
# 3,716 fps = 269.1 us), not the graph-API default (634 fps = 1,577 us). Quoting
# the default here would be quoting the pathology twice — panel E already covers
# it, with kernel-level evidence rather than an end-to-end number.
F2 = [
    ("Mojo (1 stream)",                50.6,   C["mojo"]),
    ("ONNX-RT + TensorRT",             61.0,   C["trt"]),
    ("ONNX Runtime",                  126.0,   C["ort"]),
    ("PyTorch (CUDA)",                168.5,   C["torch"]),
    ("MAX Engine (tuned graph)",      269.1,   C["max"]),
    ("JAX",                           327.5,   C["jax"]),
]


def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    labels = [r[0] for r in F2][::-1]
    vals = [r[1] for r in F2][::-1]
    cols = [r[2] for r in F2][::-1]
    ys = range(len(vals))

    ax.barh(list(ys), vals, color=cols, height=0.66, zorder=3)

    for y, v in zip(ys, vals):
        ax.text(v + 6, y, f"{v:,.1f} μs", va="center", ha="left",
                fontsize=15, weight="bold", color=INK, zorder=4)

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=15.5)
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("latency per image at batch 1  (μs, lower is better)", labelpad=10)
    ax.grid(axis="x", color=RULE, lw=1.0, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig2_latency_bs1.svg")


# ============================================================== FIGURE 3 ====
# SOURCE: CNNTesting/KERNEL_COMPARISON.md, 2026-07-03 Nsight Systems traces.
STAGES = [
    # Merged into six groups: nine legend entries needed three legend rows,
    # which cost more height than the detail was worth at this size. Pool1 and
    # pool2 merge cleanly — MAX routes both to the same generic window-reduce.
    ("normalize + argmax",  C["mojo1"], [158,    0,    0]),   # Mojo does both on-GPU
    ("conv1 (+pool1)",      "#C8783C",  [127,  344,  149]),
    ("pools",               "#9FB0A0",  [ 62,  903,  925]),
    ("conv2",               "#3E6D9C",  [541,  608,   80]),
    ("conv3",               "#B4342A",  [390, 4101,   16]),
    ("FC + epilogues",      "#7B5EA7",  [147,   10,  501]),
]
# Two-line tick labels: a separate subtitle text collided with the tick labels
# once the axis got shorter.
F3_COLS = ["Mojo\nhand-written", "MAX — RSCF\ngraph default",
           "MAX — FCRS\ncuDNN, TF32"]
F3_TOTALS = [1425, 5966, 1671]
F3_CLIP = 1950.0
F3_TOP = 2420.0          # headroom above the clip for the total labels


def fig3():
    fig, ax = plt.subplots(figsize=(6.9, 2.95))
    xs = range(3)
    bottoms = [0.0, 0.0, 0.0]

    for name, colour, vals in STAGES:
        drawn = [min(v, max(0.0, F3_CLIP - bottoms[i])) for i, v in enumerate(vals)]
        ax.bar(list(xs), drawn, bottom=bottoms, color=colour, width=0.58,
               label=name, zorder=3, edgecolor=PAPER, linewidth=1.1)
        for i, v in enumerate(vals):
            if drawn[i] > 190:
                ax.text(i, bottoms[i] + drawn[i] / 2, f"{v:,}", ha="center",
                        va="center", fontsize=12.5, color="white", weight="bold",
                        zorder=4)
            bottoms[i] += v

    ax.set_ylim(0, F3_TOP)
    ax.set_yticks([0, 500, 1000, 1500, 1950])
    ax.set_yticklabels(["0", "500", "1,000", "1,500", "1,950"])
    ax.set_xlim(-0.62, 2.62)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(F3_COLS, fontsize=13.5)
    ax.get_xticklabels()[0].set_weight("bold")
    ax.set_ylabel("GPU kernel time / image  (ns)", labelpad=8)
    ax.grid(axis="y", color=RULE, lw=1.0, alpha=0.7)
    ax.set_axisbelow(True)

    # RSCF runs off the top. Marked three ways, because a dashed line alone
    # reads as decoration: a torn edge on the bar, a break glyph on the y-axis,
    # and the truncation value as the final tick.
    import numpy as np
    x0, x1 = 1 - 0.29, 1 + 0.29
    zx = np.linspace(x0, x1, 15)
    zy = F3_CLIP + np.where(np.arange(15) % 2 == 0, -42, 42)
    ax.fill_between(zx, zy, F3_CLIP + 200, color=PAPER, zorder=5, lw=0)
    ax.plot(zx, zy, color=INK, lw=2.2, zorder=6, solid_joinstyle="miter")

    yf = F3_CLIP / F3_TOP
    for off in (-0.021, 0.021):
        ax.plot([-0.026, 0.026], [yf + off - 0.021, yf + off + 0.021],
                transform=ax.transAxes, color=INK, lw=2.0, clip_on=False,
                zorder=10)

    # Totals sit above each bar; RSCF's carries the reason it is off-scale.
    for i, t in enumerate(F3_TOTALS):
        # clipped bars need clearance above the torn edge, and the label must
        # sit above the fill that masks it
        y = min(t, F3_CLIP) + (255 if t > F3_CLIP else 90)
        ax.text(i, y, f"{t:,} ns", ha="center", fontsize=15, weight="bold",
                color=INK, zorder=8)
    # The "why it is truncated" note lives in the poster caption, not here —
    # in-figure it collided with the total label at this size.

    # Legend below in three columns — the poster column is short on height.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              frameon=False, fontsize=12.5, columnspacing=1.2,
              labelspacing=0.30, handlelength=1.3, handletextpad=0.5)
    save(fig, "fig3_kernel_scoreboard.svg")


# ============================================================== FIGURE 4 ====
# SOURCE: LeNet5Mojo/TODO.md:552 stream re-sweep, batch 100, clocks locked.
F4_X = [1, 5, 8, 12, 16]
F4_Y = [None, 1.20e6, 1.30e6, 1.42e6, 1.42e6]


def fig4():
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    xs = F4_X[1:]
    ys = F4_Y[1:]
    ax.plot(xs, ys, color=C["mojo"], marker="o", markersize=11, lw=3.4,
            markeredgecolor=PAPER, markeredgewidth=1.6, zorder=4)
    ax.axvline(12, color=MUTED, lw=1.5, ls=(0, (4, 4)), zorder=2)
    ax.annotate("knee at 12 — was 5,\nbefore conv3 became a GEMM",
                xy=(12, 1.42e6), xytext=(4.4, 1.53e6),
                fontsize=13, color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.7,
                                shrinkA=4, shrinkB=8))
    for x, y in zip(xs, ys):
        ax.text(x, y - 4.5e4, f"{y/1e6:.2f}M", ha="center", va="top",
                fontsize=12, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xlim(4, 17)
    ax.set_ylim(1.08e6, 1.60e6)
    ax.set_yticks([1.1e6, 1.2e6, 1.3e6, 1.4e6, 1.5e6])
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_xlabel("CUDA streams  (batch 100)", labelpad=8)
    ax.set_ylabel("throughput (img/s)", labelpad=8)
    ax.grid(axis="y", color=RULE, lw=1.0, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig4_stream_knee.svg")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
