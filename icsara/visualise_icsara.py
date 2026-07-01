"""
visualise_icsara.py
-------------------
Produces four publication-ready visualisations from icsara_classified.csv:

  1. icsara_distribution_{slug}.png      — item count per topic (bar chart)
  2. icsara_comparison_{slug}.png        — side-by-side ICSARA vs addenda (grouped bar)
  3. icsara_gap_{slug}.png               — gap analysis: ICSARA % minus addenda %
  4. icsara_section_heatmap_{slug}.png   — per-section topic breakdown (heatmap)

Style matches visualise_topics.py exactly.
ICSARA accent colour: #2CA02C (muted green) vs addenda blue #4C72B0.

Usage:
    python visualise_icsara.py

    python visualise_icsara.py \\
        --classified  icsara_classified.csv \\
        --addenda_dist addenda_topic_counts.csv \\
        --output_dir  C:/Users/olesc/PycharmProjects/thesis/icsara \\
        --round_label "ICSARA Round 2"

Addenda distribution input (--addenda_dist):
    A two-column CSV with headers: topic_code, pct
    e.g.  T00,7.6
          T01,8.2  ...
    If not supplied, the comparison and gap charts are skipped.

Dependencies:
    pip install pandas matplotlib seaborn numpy
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ── Shared style (mirrors visualise_topics.py exactly) ────────────────────────

FONT_FAMILY    = "DejaVu Sans"
COLOR_BAR      = "#009E73"   # ICSARA accent — bluish green (Okabe-Ito)
COLOR_ADDENDA  = "#0072B2"   # adenda blue (Okabe-Ito)
COLOR_POSITIVE = "#E69F00"   # orange (Okabe-Ito) for positive gaps
COLOR_NEGATIVE = "#0072B2"   # blue (Okabe-Ito) for negative gaps

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

# ── Topic labels per round ────────────────────────────────────────────────────

TOPIC_LABELS_R1 = {
    "T00": "T00  Native forest & vegetation",
    "T01": "T01  Flora monitoring & species conservation",
    "T02": "T02  Drainage & hydrology",
    "T03": "T03  Water infrastructure & surface works",
    "T04": "T04  Processing plant & infrastructure",
    "T05": "T05  Water process & chemical treatment",
    "T06": "T06  Environmental impact & measures",
    "T07": "T07  Hazardous waste & emergency planning",
    "T08": "T08  Project phases (construction/operation/closure)",
    "T09": "T09  Soil, revegetation & disposal",
    "T10": "T10  Fauna baseline & sampling",
    "T11": "T11  Air quality & health risk",
    "T12": "T12  Fauna relocation & monitoring",
    "T13": "T13  Indigenous communities (GHPPI)",
}

TOPIC_LABELS_R2 = {
    "T00": "T00  Native forest & vegetation",
    "T01": "T01  Water process & chemical treatment",
    "T02": "T02  Air quality & emissions",
    "T03": "T03  Project phases & drainage",
    "T04": "T04  Processing plant & infrastructure",
    "T05": "T05  Indigenous communities (GHPPI)",
    "T06": "T06  Fauna relocation & conservation",
    "T07": "T07  Soil, emergency & drainage",
    "T08": "T08  Flora monitoring & planting",
    "T09": "T09  Environmental planning & measures",
    "T10": "T10  Health risk assessment",
    "T11": "T11  Hydrology & water characterisation",
    "T12": "T12  Air quality & health risk",
}

# Round 3 topic labels (k=12, final model k12v2)
TOPIC_LABELS_R3 = {
    "T00": "T00  Project phases & environmental impact",
    "T01": "T01  Flora species & conservation planning",
    "T02": "T02  Air quality & emissions monitoring",
    "T03": "T03  Fauna relocation & baseline sampling",
    "T04": "T04  Water process & chemical treatment",
    "T05": "T05  Health risk assessment",
    "T06": "T06  Indigenous communities (GHPPI)",
    "T07": "T07  Project phases & emissions (operation/closure)",
    "T08": "T08  Native forest & species conservation",
    "T09": "T09  Emergency planning & hazardous waste",
    "T10": "T10  Drainage & hydrology",
    "T11": "T11  Soil, vegetation & revegetation",
}

ROUND_LABELS = {
    "r1": TOPIC_LABELS_R1,
    "r2": TOPIC_LABELS_R2,
    "r3": TOPIC_LABELS_R3,
}

ROUND_TOPIC_ORDER = {
    "r1": [f"T{i:02d}" for i in range(14)],
    "r2": [f"T{i:02d}" for i in range(13)],
    "r3": [f"T{i:02d}" for i in range(12)],
}

# Active labels and order — set by --round argument in main()
TOPIC_LABELS = TOPIC_LABELS_R2
TOPIC_ORDER  = ROUND_TOPIC_ORDER["r2"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def wrap_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(label, width))


def topic_counts(df: pd.DataFrame) -> pd.Series:
    """Return item counts per topic code, all 15 topics present (0 if missing)."""
    counts = df[df["item_number"].notna()]["topic_code"].value_counts()
    return counts.reindex(TOPIC_ORDER, fill_value=0)


# ── 1. Distribution bar chart ──────────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, round_label: str, out_path: Path) -> None:
    counts = topic_counts(df)
    n      = len(counts)
    labels = [wrap_label(TOPIC_LABELS.get(c, c)) for c in counts.index]
    pct    = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(13, 0.55 * n + 2))

    ax.barh(range(n), counts.values, color=COLOR_BAR,
            edgecolor="white", linewidth=0.5, height=0.7)

    for i, (v, p) in enumerate(zip(counts.values, pct)):
        ax.text(v + counts.max() * 0.01, i, f"{p:.1f}%",
                va="center", ha="left", fontsize=8, color="#555555")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of items")
    ax.set_title(f"Topic distribution — {round_label}\n"
                 f"({int(counts.sum())} total items, {n} topics)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.margins(x=0.12)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 2. Side-by-side comparison bar chart ──────────────────────────────────────

def plot_comparison(df: pd.DataFrame,
                    addenda_pct: pd.Series,
                    round_label: str,
                    out_path: Path) -> None:
    counts      = topic_counts(df)
    icsara_pct  = (counts / counts.sum() * 100).reindex(TOPIC_ORDER, fill_value=0)
    addenda_pct = addenda_pct.reindex(TOPIC_ORDER, fill_value=0)

    n      = len(TOPIC_ORDER)
    labels = [wrap_label(TOPIC_LABELS.get(c, c)) for c in TOPIC_ORDER]
    y      = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(13, 0.7 * n + 2))

    ax.barh(y + height / 2, icsara_pct.values,  height, color=COLOR_BAR,
            edgecolor="white", linewidth=0.4, label="ICSARA items")
    ax.barh(y - height / 2, addenda_pct.values, height, color=COLOR_ADDENDA,
            edgecolor="white", linewidth=0.4, label="Adenda chunks")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of total")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.margins(x=0.08)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.subplots_adjust(left=0.28)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 3. Gap analysis chart ──────────────────────────────────────────────────────

def plot_gap(df: pd.DataFrame,
             addenda_pct: pd.Series,
             round_label: str,
             out_path: Path) -> None:
    counts      = topic_counts(df)
    icsara_pct  = (counts / counts.sum() * 100).reindex(TOPIC_ORDER, fill_value=0)
    addenda_pct = addenda_pct.reindex(TOPIC_ORDER, fill_value=0)
    gap         = (icsara_pct - addenda_pct).reindex(TOPIC_ORDER)

    # Sort by gap value for readability
    gap_sorted  = gap.sort_values()
    labels      = [wrap_label(TOPIC_LABELS.get(c, c)) for c in gap_sorted.index]
    colors      = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE
                   for v in gap_sorted.values]

    n   = len(gap_sorted)
    fig, ax = plt.subplots(figsize=(13, 0.55 * n + 2))

    ax.barh(range(n), gap_sorted.values, color=colors,
            edgecolor="white", linewidth=0.4, height=0.7)

    # Zero line
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")

    # Value labels
    for i, v in enumerate(gap_sorted.values):
        ha  = "left"  if v >= 0 else "right"
        off = 0.15    if v >= 0 else -0.15
        ax.text(v + off, i, f"{v:+.1f}pp",
                va="center", ha=ha, fontsize=8, color="#555555")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Percentage point difference (ICSARA % − Adenda %)")
    ax.set_title(f"ICSARA concern gap — {round_label}\n"
                 f"(positive = ICSARA raised more than adenda addressed; "
                 f"negative = adenda addressed more than ICSARA raised)")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:+.0f}pp"))
    ax.margins(x=0.15)

    pos_patch = mpatches.Patch(color=COLOR_POSITIVE, label="ICSARA > Adenda (under-addressed)")
    neg_patch = mpatches.Patch(color=COLOR_NEGATIVE, label="Adenda > ICSARA (over-addressed)")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 4. Per-section heatmap ─────────────────────────────────────────────────────

def plot_section_heatmap(df: pd.DataFrame,
                         round_label: str,
                         out_path: Path) -> None:
    items = df[df["item_number"].notna()].copy()

    # Shorten section labels for display
    def shorten_section(s: str) -> str:
        # Keep Roman numeral prefix + first few words
        s = s.strip()
        parts = s.split(".")
        if len(parts) >= 2:
            roman  = parts[0].strip()
            rest   = ".".join(parts[1:]).strip()
            words  = rest.split()[:5]
            return f"{roman}. {' '.join(words)}"
        return s[:40]

    items["section_short"] = items["section"].apply(shorten_section)

    # Build count matrix: sections × topics
    matrix = (items.groupby(["section_short", "topic_code"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=TOPIC_ORDER, fill_value=0))

    # Sort sections by Roman numeral order (PREAMBLE last)
    def section_sort_key(s):
        roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
                     "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
                     "XI": 11, "XII": 12, "XIII": 13, "XIV": 14,
                     "XV": 15, "XVI": 16, "XVII": 17}
        prefix = s.split(".")[0].strip()
        return roman_map.get(prefix, 99)

    matrix = matrix.loc[sorted(matrix.index, key=section_sort_key)]

    # Normalise to row proportions for colour scaling
    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix_norm = matrix.div(row_sums, axis=0)

    col_labels = [wrap_label(TOPIC_LABELS.get(c, c), width=18)
                  for c in matrix.columns]

    n_sections = len(matrix)
    n_topics   = len(TOPIC_ORDER)
    fig_h      = max(6, n_sections * 0.55 + 3)
    fig_w      = max(14, n_topics  * 0.85 + 4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Annotate with raw counts
    sns.heatmap(
        matrix_norm.values,
        ax=ax,
        cmap="YlGn",          # green ramp to match ICSARA accent
        vmin=0,
        annot=matrix.values,
        fmt="d",
        annot_kws={"size": 8},
        xticklabels=col_labels,
        yticklabels=matrix.index.tolist(),
        linewidths=0.3,
        linecolor="#eeeeee",
        cbar_kws={"label": "Proportion of section items", "shrink": 0.6},
    )

    ax.set_title(f"ICSARA items by section and topic — {round_label}\n"
                 f"(colour = row proportion; numbers = item count)",
                 pad=14)
    ax.set_xlabel("Topic")
    ax.set_ylabel("ICSARA section")
    ax.tick_params(axis="x", rotation=45, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

ALL_ROUNDS = {
    "r1": {
        "classified":  BASE / "icsara" / "data" / "icsara_classified_round_1.csv",
        "output_dir":  BASE / "icsara",
        "round_label": "ICSARA Round 1",
        "round_key":   "r1",
    },
    "r2": {
        "classified":  BASE / "icsara" / "data" / "icsara_classified_round_2.csv",
        "output_dir":  BASE / "icsara",
        "round_label": "ICSARA Round 2",
        "round_key":   "r2",
    },
    "r3": {
        "classified":  BASE / "icsara" / "data" / "icsara_classified_round_3.csv",
        "output_dir":  BASE / "icsara",
        "round_label": "ICSARA Round 3",
        "round_key":   "r3",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualise ICSARA classification results — all rounds."
    )
    parser.add_argument("--round", default="all", choices=["r1", "r2", "r3", "all"],
        help="Which round to run (default: all)")
    args = parser.parse_args()

    rounds = ALL_ROUNDS if args.round == "all" else {args.round: ALL_ROUNDS[args.round]}

    for rkey, cfg in rounds.items():
        print(f"\n{'='*50}")
        print(f"Processing {cfg['round_label']}...")

        out = Path(cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

        df   = pd.read_csv(cfg["classified"])
        items = df[df["item_number"].notna()]
        print(f"  {len(items)} items")

        global TOPIC_LABELS, TOPIC_ORDER
        TOPIC_LABELS = ROUND_LABELS.get(rkey, TOPIC_LABELS_R2)
        TOPIC_ORDER  = ROUND_TOPIC_ORDER.get(rkey, ROUND_TOPIC_ORDER["r2"])

        slug = cfg["round_label"].lower().replace(" ", "_")

        ADENDA_DIST = {
            "r1": pd.Series({
                "T00": 5.2, "T01": 5.5, "T02": 4.4, "T03": 10.6,
                "T04": 2.4, "T05": 5.4, "T06": 8.6, "T07": 8.2,
                "T08": 10.9,"T09": 8.3, "T10": 7.0, "T11": 7.4,
                "T12": 9.0, "T13": 7.1,
            }),
            "r2": pd.Series({
                "T00": 5.1, "T01": 9.5, "T02": 7.9, "T03": 11.9,
                "T04": 7.4, "T05": 3.4, "T06": 8.7, "T07": 8.2,
                "T08": 7.4, "T09": 10.2,"T10": 2.8, "T11": 13.8,
                "T12": 3.6,
            }),
            "r3": pd.Series({
                "T00": 13.3,"T01": 4.0, "T02": 5.9, "T03": 11.5,
                "T04": 10.2,"T05": 4.9, "T06": 9.0, "T07": 8.4,
                "T08": 8.5, "T09": 8.5, "T10": 8.9, "T11": 8.2,
            }),
        }
        addenda_pct = ADENDA_DIST[rkey]

        plot_distribution(df, cfg["round_label"],
                          out / f"icsara_distribution_{slug}.png")
        plot_comparison(df, addenda_pct, cfg["round_label"],
                        out / f"icsara_comparison_{slug}.png")
        plot_gap(df, addenda_pct, cfg["round_label"],
                 out / f"icsara_gap_{slug}.png")
        plot_section_heatmap(df, cfg["round_label"],
                             out / f"icsara_section_heatmap_{slug}.png")

    print("\nDone. All rounds visualised.")


if __name__ == "__main__":
    main()