"""
visualise_crossround.py
-----------------------
Produces three cross-round comparison visualisations using the master
topic codebook to align Round 1, 2 and 3 ICSARA and adenda data.

Charts:
  1. crossround_icsara_items.png     — ICSARA item counts by master topic per round
  2. crossround_coverage.png         — mean coverage score by master topic per round
  3. crossround_combined.png         — ICSARA % + coverage score combined per round

Usage:
    python visualise_crossround.py

Dependencies:
    pip install pandas matplotlib numpy
"""

import numpy as np
import pandas as pd
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

CLASSIFIED = {
    "r1": BASE / "icsara" / "data" / "icsara_classified_round_1.csv",
    "r2": BASE / "icsara" / "data" / "icsara_classified_round_2.csv",
    "r3": BASE / "icsara" / "data" / "icsara_classified_round_3.csv",
}

COVERAGE = {
    "r1": BASE / "icsara" / "data" / "coverage_round_1.csv",
    "r2": BASE / "icsara" / "data" / "coverage_round_2.csv",
    "r3": BASE / "icsara" / "data" / "coverage_round_3.csv",
}

OUTPUT_DIR = BASE / "icsara"

# ── Master codebook — single assignment per round topic ────────────────────────
# Each round topic maps to exactly one master topic

MASTER_LABELS = {
    "M01": "Water & drainage",
    "M02": "Native vegetation & flora",
    "M03": "Fauna & wildlife",
    "M04": "Air quality & health risk",
    "M05": "Indigenous communities (GHPPI)",
    "M06": "Emergency & hazardous waste",
    "M07": "Project phases & infrastructure",
    "M08": "Soil & revegetation",
    "M09": "Environmental planning & measures",
}

MASTER_ORDER = ["M01","M02","M03","M04","M05","M06","M07","M08","M09"]

# Single mapping: round topic → master topic
TOPIC_TO_MASTER = {
    # Round 1
    "r1": {
        "T00": "M02", "T01": "M02",
        "T02": "M01", "T03": "M01", "T05": "M01",
        "T04": "M07", "T08": "M07",
        "T06": "M09",
        "T07": "M06",
        "T09": "M08",
        "T10": "M03", "T12": "M03",
        "T11": "M04",
        "T13": "M05",
    },
    # Round 2 — T03 → M01, T07 → M06 (resolved duplicates)
    "r2": {
        "T00": "M02", "T08": "M02",
        "T01": "M01", "T03": "M01", "T11": "M01",
        "T04": "M07",
        "T05": "M05",
        "T06": "M03",
        "T07": "M06",
        "T02": "M04", "T10": "M04", "T12": "M04",
        "T09": "M09",
    },
    # Round 3
    "r3": {
        "T01": "M02", "T08": "M02",
        "T04": "M01", "T10": "M01",
        "T03": "M03",
        "T02": "M04", "T05": "M04",
        "T06": "M05",
        "T09": "M06",
        "T00": "M07", "T07": "M07",
        "T11": "M08",
    },
}

# ── Style ──────────────────────────────────────────────────────────────────────

FONT_FAMILY = "DejaVu Sans"
COLORS = {
    "r1": "#0072B2",   # blue (Okabe-Ito)
    "r2": "#009E73",   # bluish green (Okabe-Ito)
    "r3": "#E69F00",   # orange (Okabe-Ito)
}
ROUND_LABELS = {"r1": "Review round 1", "r2": "Review round 2", "r3": "Review round 3"}

plt.rcParams.update({
    "font.family":        FONT_FAMILY,
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})


def wrap_label(label, width=24):
    return "\n".join(textwrap.wrap(label, width))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    """Load classified and coverage CSVs, map to master topics."""
    classified = {}
    coverage   = {}

    for rkey in ["r1", "r2", "r3"]:
        df = pd.read_csv(CLASSIFIED[rkey])
        df = df[df["item_number"].notna()].copy()
        df["master"] = df["topic_code"].map(TOPIC_TO_MASTER[rkey])
        classified[rkey] = df

        cov = pd.read_csv(COVERAGE[rkey])
        cov = cov[cov["item_number"].notna()].copy()
        cov["master"] = cov["topic_code"].map(TOPIC_TO_MASTER[rkey])
        coverage[rkey] = cov

    return classified, coverage


# ── 1. ICSARA item counts by master topic ──────────────────────────────────────

def plot_icsara_items(classified, out_path):
    n       = len(MASTER_ORDER)
    y       = np.arange(n)
    height  = 0.25
    labels  = [wrap_label(MASTER_LABELS[m]) for m in MASTER_ORDER]

    fig, ax = plt.subplots(figsize=(13, 0.75 * n + 2))

    for i, rkey in enumerate(["r1", "r2", "r3"]):
        df     = classified[rkey]
        total  = len(df)
        counts = df.groupby("master").size().reindex(MASTER_ORDER, fill_value=0)
        pcts   = counts / total * 100
        offset = (i - 1) * height
        bars   = ax.barh(y + offset, pcts.values, height,
                         color=COLORS[rkey], label=ROUND_LABELS[rkey],
                         edgecolor="white", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of round's ICSARA items")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.7)
    ax.margins(x=0.06)
    # Note missing Round 2 bar for Soil & revegetation
    fig.text(0.5, 0.01,
             "* Soil & revegetation (M08) had no distinct Round 2 topic in the master codebook mapping.",
             ha="center", fontsize=7.5, color="#888780", style="italic")
    fig.tight_layout()
    fig.subplots_adjust(left=0.26, bottom=0.06)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 1b. Poster version of ICSARA items chart ──────────────────────────────────

def plot_icsara_items_poster(classified, out_path):
    """Poster-optimised version — top 5 topics by total ICSARA volume, larger fonts."""
    totals = pd.Series(0.0, index=MASTER_ORDER)
    for rkey in ["r1", "r2", "r3"]:
        df     = classified[rkey]
        total  = len(df)
        counts = df.groupby("master").size().reindex(MASTER_ORDER, fill_value=0)
        totals += counts / total * 100

    top_topics = totals.nlargest(5).index.tolist()
    top_topics = sorted(top_topics, key=lambda m: totals[m], reverse=False)

    n      = len(top_topics)
    y      = np.arange(n)
    height = 0.25
    labels = [wrap_label(MASTER_LABELS[m], width=20) for m in top_topics]

    fig, ax = plt.subplots(figsize=(11, 0.9 * n + 2))

    for i, rkey in enumerate(["r1", "r2", "r3"]):
        df     = classified[rkey]
        total  = len(df)
        counts = df.groupby("master").size().reindex(MASTER_ORDER, fill_value=0)
        pcts   = counts / total * 100
        vals   = [pcts.get(m, 0) for m in top_topics]
        offset = (i - 1) * height
        ax.barh(y + offset, vals, height,
                color=COLORS[rkey], label=ROUND_LABELS[rkey],
                edgecolor="white", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("% of round's ICSARA items", fontsize=11)
    ax.set_title("ICSARA concerns by master topic — cross-round comparison",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    ax.margins(x=0.06)
    fig.tight_layout()
    fig.subplots_adjust(left=0.28)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 2. Mean coverage score by master topic ─────────────────────────────────────

def plot_coverage(coverage, out_path):
    n      = len(MASTER_ORDER)
    y      = np.arange(n)
    height = 0.25
    labels = [wrap_label(MASTER_LABELS[m]) for m in MASTER_ORDER]

    fig, ax = plt.subplots(figsize=(12, 0.75 * n + 2))

    for i, rkey in enumerate(["r1", "r2", "r3"]):
        df    = coverage[rkey].dropna(subset=["coverage_score", "master"])
        means = df.groupby("master")["coverage_score"].mean().reindex(MASTER_ORDER)
        offset = (i - 1) * height
        ax.barh(y + offset, means.values, height,
                color=COLORS[rkey], label=ROUND_LABELS[rkey],
                edgecolor="white", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean coverage score (1 = not addressed, 3 = fully addressed)")
    ax.set_xlim(1, 3)
    ax.axvline(2, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.7)
    fig.tight_layout()
    fig.subplots_adjust(left=0.26)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 3. Combined: ICSARA % as bar, coverage as scatter ─────────────────────────

def plot_combined(classified, coverage, out_path):
    n      = len(MASTER_ORDER)
    y      = np.arange(n)
    height = 0.25
    labels = [wrap_label(MASTER_LABELS[m]) for m in MASTER_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 0.75 * n + 2),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: ICSARA item %
    for i, rkey in enumerate(["r1", "r2", "r3"]):
        df     = classified[rkey]
        total  = len(df)
        counts = df.groupby("master").size().reindex(MASTER_ORDER, fill_value=0)
        pcts   = counts / total * 100
        offset = (i - 1) * height
        ax1.barh(y + offset, pcts.values, height,
                 color=COLORS[rkey], label=ROUND_LABELS[rkey],
                 edgecolor="white", linewidth=0.3)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("% of ICSARA items")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.7)
    ax1.margins(x=0.06)

    # Right: mean coverage score
    for i, rkey in enumerate(["r1", "r2", "r3"]):
        df    = coverage[rkey].dropna(subset=["coverage_score", "master"])
        means = df.groupby("master")["coverage_score"].mean().reindex(MASTER_ORDER)
        offset = (i - 1) * height
        ax2.barh(y + offset, means.values, height,
                 color=COLORS[rkey], edgecolor="white", linewidth=0.3)

    ax2.set_yticks(y)
    ax2.set_yticklabels([], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean coverage score")
    ax2.set_xlim(1, 3)
    ax2.axvline(2, color="#cccccc", linewidth=0.8, linestyle="--")
    ax2.margins(x=0.06)

    fig.tight_layout()
    fig.subplots_adjust(left=0.22, wspace=0.05)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 4. Cross-round gap chart ───────────────────────────────────────────────────

def compute_adenda_master_pct(round_key: str) -> pd.Series:
    """Compute adenda % per master topic from the chunks_topics CSV."""
    chunks_lda_paths = {
        "r1": BASE / "adenda_r1" / "chunks_topics_r1_k14v2.csv",
        "r2": BASE / "adenda_r2" / "chunks_topics_r2_k13v1.csv",
        "r3": BASE / "adenda_r3" / "chunks_topics_r3_k12v2.csv",
    }
    df = pd.read_csv(chunks_lda_paths[round_key], usecols=["dominant_topic"])
    df["master"] = df["dominant_topic"].apply(
        lambda t: TOPIC_TO_MASTER[round_key].get(f"T{int(t):02d}", None)
    )
    df = df.dropna(subset=["master"])
    counts = df.groupby("master").size().reindex(MASTER_ORDER, fill_value=0)
    return counts / counts.sum() * 100


def plot_crossround_gap(classified, out_path):
    """
    Diverging grouped bar chart: ICSARA % minus adenda % per master topic per round.
    Positive = ICSARA raised more than adenda addressed (under-addressed).
    Negative = adenda addressed more than ICSARA raised (over-addressed).
    Round colour used consistently regardless of direction.
    Water & drainage (M01) and Fauna & wildlife (M03) highlighted.
    """
    n      = len(MASTER_ORDER)
    y      = np.arange(n)
    height = 0.25
    labels = [wrap_label(MASTER_LABELS[m]) for m in MASTER_ORDER]

    fig, ax = plt.subplots(figsize=(13, 0.75 * n + 2))

    # Highlight bands for key topics
    highlight_topics = {"M01", "M03"}
    for j, m in enumerate(MASTER_ORDER):
        if m in highlight_topics:
            ax.axhspan(j - 0.48, j + 0.48, color="#f5f5f0", zorder=0)

    for i, rkey in enumerate(["r1", "r2", "r3"]):
        df         = classified[rkey]
        total      = len(df)
        counts     = df.groupby("master").size().reindex(MASTER_ORDER, fill_value=0)
        icsara_pct = counts / total * 100
        adenda_pct = compute_adenda_master_pct(rkey)
        gap        = icsara_pct - adenda_pct

        offset = (i - 1) * height
        ax.barh(y + offset, gap.values, height,
                color=COLORS[rkey], label=ROUND_LABELS[rkey],
                edgecolor="white", linewidth=0.3, zorder=2)

        # Print gap values for reference
        print(f"\n  {ROUND_LABELS[rkey]} gaps:")
        for m, v in zip(MASTER_ORDER, gap.values):
            print(f"    {m}: {v:+.1f}pp")

    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage point difference (ICSARA % − Adenda %)")
    round_ns = {"r1": 394, "r2": 205, "r3": 114}
    handles, lbls = ax.get_legend_handles_labels()
    lbls = [f"{l} (n={round_ns[r]})" for l, r in zip(lbls, ["r1","r2","r3"])]
    ax.legend(handles, lbls, loc="lower right", fontsize=9, framealpha=0.7)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.subplots_adjust(left=0.26)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 5. Poster-optimised gap chart ─────────────────────────────────────────────

def plot_crossround_gap_poster(classified, out_path):
    """
    Poster version of the cross-round gap chart.
    Shows top 5 topics, highlights Water & drainage and Fauna & wildlife rows,
    adds value labels on the two key bars, updated terminology.
    """
    n_show = 5

    gap_data = {}
    for rkey in ["r1", "r2", "r3"]:
        df         = classified[rkey]
        total      = len(df)
        counts     = df.groupby("master").size().reindex(MASTER_ORDER, fill_value=0)
        icsara_pct = counts / total * 100
        adenda_pct = compute_adenda_master_pct(rkey)
        gap_data[rkey] = (icsara_pct - adenda_pct).reindex(MASTER_ORDER)

    max_abs_gap = pd.DataFrame(gap_data).abs().max(axis=1)
    top_topics  = max_abs_gap.nlargest(n_show).index.tolist()
    top_topics  = sorted(top_topics, key=lambda m: gap_data["r3"][m], reverse=False)

    n      = len(top_topics)
    y      = np.arange(n)
    height = 0.25
    labels = [wrap_label(MASTER_LABELS[m], width=20) for m in top_topics]

    fig, ax = plt.subplots(figsize=(12, 0.9 * n + 2))

    # Highlight bands for Water & drainage (M01) and Fauna & wildlife (M03)
    highlight_topics = {"M01", "M03"}
    for j, m in enumerate(top_topics):
        if m in highlight_topics:
            ax.axhspan(j - 0.48, j + 0.48, color="#f5f5f0", zorder=0)

    for i, rkey in enumerate(["r1", "r2", "r3"]):
        gaps   = [gap_data[rkey][m] for m in top_topics]
        offset = (i - 1) * height
        bars = ax.barh(y + offset, gaps, height,
                       color=COLORS[rkey], label=ROUND_LABELS[rkey],
                       edgecolor="white", linewidth=0.4, zorder=2)

        # Value labels on key bars only
        for j, (m, gap) in enumerate(zip(top_topics, gaps)):
            if m == "M01" and rkey == "r1" and gap > 0:
                ax.text(gap + 0.3, j + offset, f"+{gap:.1f}pp",
                        va="center", ha="left", fontsize=9.5,
                        fontweight="bold", color=COLORS[rkey])
            if m == "M03" and rkey == "r3" and gap > 0:
                ax.text(gap + 0.3, j + offset, f"+{gap:.1f}pp",
                        va="center", ha="left", fontsize=9.5,
                        fontweight="bold", color=COLORS[rkey])

    ax.axvline(0, color="#444444", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage point difference (ICSARA % − Adenda %)\n"
                  "(positive = under-represented in adenda; negative = over-represented)",
                  fontsize=10)
    ax.set_title("ICSARA concern representation gap — cross-round comparison",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    ax.margins(x=0.14)
    fig.tight_layout()
    fig.subplots_adjust(left=0.28)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classified, coverage = load_data()

    plot_icsara_items(classified,
                      OUTPUT_DIR / "crossround_icsara_items.png")
    plot_icsara_items_poster(classified,
                             OUTPUT_DIR / "crossround_icsara_items_poster.png")
    plot_coverage(coverage,
                  OUTPUT_DIR / "crossround_coverage.png")
    plot_combined(classified, coverage,
                  OUTPUT_DIR / "crossround_combined.png")
    plot_crossround_gap(classified,
                        OUTPUT_DIR / "crossround_gap.png")
    plot_crossround_gap_poster(classified,
                               OUTPUT_DIR / "crossround_gap_poster.png")

    print("\nDone. Five PNGs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()