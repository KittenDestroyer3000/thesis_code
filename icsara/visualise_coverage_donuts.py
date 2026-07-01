"""
visualise_coverage_donuts.py
----------------------------
Three separate donut PNGs, one per review round.
Output: coverage_donut_r1/r2/r3.png in icsara/
"""

import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

COVERAGE_FILES = {
    "r1": BASE / "icsara" / "data" / "coverage_round_1.csv",
    "r2": BASE / "icsara" / "data" / "coverage_round_2.csv",
    "r3": BASE / "icsara" / "data" / "coverage_round_3.csv",
}
OUTPUT_DIR   = BASE / "icsara"
ROUND_LABELS = {"r1": "Review round 1", "r2": "Review round 2", "r3": "Review round 3"}
COLORS       = ["#D55E00", "#F0E442", "#009E73"]
SCORE_LABELS = ["Not addressed", "Partially addressed", "Fully addressed"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "savefig.facecolor": "white",
    "savefig.dpi":       200,
    "figure.dpi":        200,
})


def make_donut(rkey, path):
    df    = pd.read_csv(path)
    items = df[df["item_number"].notna()]
    total = len(items)
    counts = items["coverage_score"].value_counts().sort_index()
    values = [int(counts.get(s, 0)) for s in [1, 2, 3]]

    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    fig.subplots_adjust(top=0.88, bottom=0.14)

    wedges, _ = ax.pie(
        values,
        colors=COLORS,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.46, edgecolor="white", linewidth=2.5),
    )

    # Centre text
    ax.text(0, 0.10, str(total), ha="center", va="center",
            fontsize=22, fontweight="bold", color="#2d2c29")
    ax.text(0, -0.18, "items", ha="center", va="center",
            fontsize=11, color="#888780")

    # Percentage labels — placed at radius 0.70 at wedge midpoint
    start_angle = 90.0
    for i, val in enumerate(values):
        pct = 100 * val / total
        if pct < 3:
            start_angle -= pct * 3.6
            continue
        mid_angle = start_angle - (pct * 3.6) / 2
        r = 0.70
        x = r * math.cos(math.radians(mid_angle))
        y = r * math.sin(math.radians(mid_angle))
        tc = "#2d2c29" if COLORS[i] == "#F0E442" else "white"
        ax.text(x, y, f"{pct:.0f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=tc)
        start_angle -= pct * 3.6



    # Sharp matplotlib legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[i], edgecolor="white",
                       linewidth=0.8, label=SCORE_LABELS[i])
        for i in range(3)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=11,
        framealpha=0,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=0.7,
        handlelength=1.2,
        handletextpad=0.4,
    )

    out_path = OUTPUT_DIR / f"coverage_donut_{rkey}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rkey, path in COVERAGE_FILES.items():
        make_donut(rkey, path)
    print("Done.")


if __name__ == "__main__":
    main()