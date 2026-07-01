"""
visualise_coverage_combined.py
------------------------------
Produces a single grouped horizontal bar chart showing coverage score
distribution across all three rounds.

Three groups (Not addressed, Partially addressed, Fully addressed),
three bars per group (Round 1, Round 2, Round 3).

Output: coverage_distribution_combined.png in icsara/

Usage:
    python visualise_coverage_combined.py

Dependencies:
    pip install pandas matplotlib numpy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

COVERAGE_FILES = {
    "r1": BASE / "icsara" / "data" / "coverage_round_1.csv",
    "r2": BASE / "icsara" / "data" / "coverage_round_2.csv",
    "r3": BASE / "icsara" / "data" / "coverage_round_3.csv",
}

OUTPUT_PATH = BASE / "icsara" / "coverage_distribution_combined.png"

COLORS = {
    "r1": "#0072B2",
    "r2": "#009E73",
    "r3": "#E69F00",
}
ROUND_LABELS = {
    "r1": "Review round 1",
    "r2": "Review round 2",
    "r3": "Review round 3",
}
SCORE_LABELS = {1: "Not addressed", 2: "Partially addressed", 3: "Fully addressed"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "savefig.dpi":       200,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
})


def main():
    # Load and compute percentages
    data = {}
    for rkey, path in COVERAGE_FILES.items():
        df    = pd.read_csv(path)
        items = df[df["item_number"].notna()]
        total = len(items)
        counts = items["coverage_score"].value_counts().sort_index()
        data[rkey] = {
            score: {"n": int(counts.get(score, 0)),
                    "pct": round(100 * counts.get(score, 0) / total, 1)}
            for score in [1, 2, 3]
        }

    scores  = [1, 2, 3]
    n_scores = len(scores)
    n_rounds = 3
    height   = 0.22
    y        = np.arange(n_scores)

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, rkey in enumerate(["r1", "r2", "r3"]):
        offset = (i - 1) * height
        pcts   = [data[rkey][s]["pct"] for s in scores]
        ns     = [data[rkey][s]["n"]   for s in scores]
        bars   = ax.barh(y + offset, pcts, height,
                         color=COLORS[rkey], label=ROUND_LABELS[rkey],
                         edgecolor="white", linewidth=0.5)

        # Annotate with n=
        for j, (pct, n) in enumerate(zip(pcts, ns)):
            ax.text(pct + 0.5, y[j] + offset, f"n={n}",
                    va="center", ha="left", fontsize=8, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels([SCORE_LABELS[s] for s in scores], fontsize=10)
    ax.set_xlabel("% of round's ICSARA items")
    ax.invert_yaxis()
    ax.set_xlim(0, 85)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.7)
    ax.margins(y=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()