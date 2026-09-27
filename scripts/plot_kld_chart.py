#!/usr/bin/env python3
"""Render the README KLD chart. Requires matplotlib; no model loading."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/assets/kld-chart-data.json"
OUT = ROOT / "docs/assets/kld-chart.png"


def main():
    data = json.loads(DATA.read_text())
    orange, green, blue, gray = "#f15a2a", "#149e80", "#397edd", "#9299a2"
    ink, muted, cream = "#242329", "#696770", "#fff8f4"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig = plt.figure(figsize=(14, 12), dpi=200, facecolor=cream)
    fig.text(.075, .956, "Qwen3.8-Flash-Next · quality vs size", fontsize=25,
             fontweight="bold", color=ink)
    fig.text(.075, .925, "KLD against the bf16 teacher · lower is better · 16 prompts × 512 tokens",
             fontsize=13, color=muted)
    ax = fig.add_axes((.09, .49, .86, .36), facecolor="white")
    ax.set_title("Published comparison · kv8", loc="left", pad=18,
                 fontsize=17, fontweight="bold", color=ink)
    colors = {"sushi": orange, "ngram": "#ff9a7e", "mlx": green, "oqe": blue, "affine": gray}
    offsets = {
        "Sushi-3bpw": (12, -22), "Sushi-4bpw": (12, -23),
        "3bpw · 4-bit n-gram": (12, 17), "4bpw · 4-bit n-gram": (12, 18),
        "iQ-MLX 3.3bpw": (12, 5), "mixed-4-8bit": (12, 5),
        "oQ4e": (12, 7), "oQ5e": (-110, -24), "affine q3": (12, 0),
    }
    for series in ("sushi", "mlx", "oqe"):
        pts = [p for p in data["legacy_kv8"] if p["series"] == series]
        ax.plot([p["gib"] for p in pts], [p["kld"] for p in pts],
                color=colors[series], linewidth=2.5, alpha=.75, zorder=2)
    for p in data["legacy_kv8"]:
        color = colors[p["series"]]
        ring = p["series"] == "ngram"
        ax.scatter(p["gib"], p["kld"], s=85, facecolors="white" if ring else color,
                   edgecolors=color if ring else "white", linewidths=2, zorder=4)
        ax.annotate(f'{p["name"]} · {p["kld"]:.4f}', (p["gib"], p["kld"]),
                    xytext=offsets[p["name"]], textcoords="offset points", fontsize=10.5,
                    fontweight="bold", color=color, zorder=5,
                    bbox=dict(boxstyle="round,pad=.3", fc="white", ec="none", alpha=.94))
    ax.set(xlim=(45, 90), ylim=(.035, .22), ylabel="KLD to first EOS",
           xlabel="Loaded weights (GiB) · SSD n-gram table excluded")
    ax.set_xticks(range(45, 91, 5))
    legend = [Line2D([], [], marker="o", linestyle="none", color=c, label=n, markersize=7)
              for n, c in [("sushi (EXL3)", orange), ("mlx-serve", green),
                           ("oMLX oQe", blue), ("MLX affine q3", gray)]]
    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=10)

    bx = fig.add_axes((.22, .16, .65, .20), facecolor="white")
    fig.text(.075, .405, "New · K2.6 tuning comparison", fontsize=18,
             fontweight="bold", color=ink)
    fig.text(.075, .38, "bf16 KV · 7,186 positions to first EOS · separate from the kv8 comparison above",
             fontsize=11, color=muted)
    rows = data["bf16_kv"]
    for p in rows:
        assert p["kv_cache_format"] == "bf16" and p["positions_to_eos"] == 7186
    values = [p["mean_kld_to_eos"] for p in rows]
    bars = bx.barh(range(len(rows)), values, height=.56,
                   color=["#edb89f", orange, "#5f7893", "#9bacbb"])
    bx.set_yticks(range(len(rows)), [p["name"] for p in rows], color=ink)
    bx.invert_yaxis()
    for bar, value in zip(bars, values):
        bx.text(value + .002, bar.get_y() + bar.get_height() / 2,
                f"{value:.6f}", va="center", fontsize=12, fontweight="bold", color=ink)
    bx.set(xlim=(0, .165), xlabel="KLD to first EOS")
    improvement = (1 - values[1] / values[0]) * 100
    fig.text(.22, .092, f"T2: {improvement:.2f}% lower KLD than untuned K2.6 at the same weight size",
             fontsize=13, fontweight="bold", color=orange)
    for axis in (ax, bx):
        axis.set_axisbelow(True)
        axis.grid(axis="y" if axis == ax else "x", color="#e9e0dc", linestyle=(0, (3, 5)))
        axis.tick_params(length=0, pad=8, colors=muted)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.xaxis.label.set_color(muted)
        axis.yaxis.label.set_color(muted)
    fig.text(.075, .048, "Top: published pack comparison; hollow markers use a 4-bit n-gram table.",
             fontsize=10, color=muted)
    fig.text(.075, .029, "Bottom: matching bf16-KV evaluations. Source values and reproduction: docs/quality-kld.md.",
             fontsize=10, color=muted)
    fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
