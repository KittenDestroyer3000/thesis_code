"""
visualise_topics.py
-------------------
Produces three publication-ready visualisations from LDA output:

  1. topic_distribution.png  — chunk count per topic (bar chart)
  2. topic_doc_heatmap.png   — per-document topic proportions (heatmap)
  3. topic_cooccurrence.png  — topic co-occurrence across documents (heatmap)

Usage:
    python visualise_topics.py \\
        --chunks   chunks_topics_k15v2.csv \\
        --topics   topics_k15v2.txt \\
        --output_dir C:/Users/olesc/PycharmProjects/thesis/addenda_lda \\
        --round_label "Adenda Round 2"

Dependencies:
    pip install pandas matplotlib seaborn numpy
"""

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
COLOR_BAR    = "#0072B2"   # blue (Okabe-Ito)
COLOR_ACCENT = "#E69F00"   # orange (Okabe-Ito)

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


# ---------------------------------------------------------------------------
# Human-readable topic name overrides
# Edit these to match your final model's topic interpretations.
# Keys are topic numbers (int). If a topic number is not listed here,
# the script falls back to the top-5 words from topics_kN.txt.
# ---------------------------------------------------------------------------
# Round 2 topic labels (k=13, final model k13v1)
TOPIC_NAME_OVERRIDES_R2 = {
    0:  "Native forest & vegetation",
    1:  "Water process & chemical treatment",
    2:  "Air quality & emissions",
    3:  "Project phases & drainage",
    4:  "Processing plant & infrastructure",
    5:  "Indigenous communities (GHPPI)",
    6:  "Fauna relocation & conservation",
    7:  "Soil, emergency & drainage",
    8:  "Flora monitoring & planting",
    9:  "Environmental planning & measures",
    10: "Health risk assessment",
    11: "Hydrology & water characterisation",
    12: "Air quality & health risk",
}

# Round 1 topic labels (k=14, final model k14v2)
TOPIC_NAME_OVERRIDES_R1 = {
    0:  "Native forest & vegetation",
    1:  "Flora monitoring & species conservation",
    2:  "Drainage & hydrology",
    3:  "Water infrastructure & surface works",
    4:  "Processing plant & infrastructure",
    5:  "Water process & chemical treatment",
    6:  "Environmental impact & measures",
    7:  "Hazardous waste & emergency planning",
    8:  "Project phases (construction/operation/closure)",
    9:  "Soil, revegetation & disposal",
    10: "Fauna baseline & sampling",
    11: "Air quality & health risk",
    12: "Fauna relocation & monitoring",
    13: "Indigenous communities (GHPPI)",
}

# Round 3 topic labels (k=12, final model k12v2)
TOPIC_NAME_OVERRIDES_R3 = {
    0:  "Project phases & environmental impact",
    1:  "Flora species & conservation planning",
    2:  "Air quality & emissions monitoring",
    3:  "Fauna relocation & baseline sampling",
    4:  "Water process & chemical treatment",
    5:  "Health risk assessment",
    6:  "Indigenous communities (GHPPI)",
    7:  "Project phases & emissions (operation/closure)",
    8:  "Native forest & species conservation",
    9:  "Emergency planning & hazardous waste",
    10: "Drainage & hydrology",
    11: "Soil, vegetation & revegetation",
}

ROUND_OVERRIDES = {
    "r1": TOPIC_NAME_OVERRIDES_R1,
    "r2": TOPIC_NAME_OVERRIDES_R2,
    "r3": TOPIC_NAME_OVERRIDES_R3,
}

# Default (backwards compatible)
TOPIC_NAME_OVERRIDES = TOPIC_NAME_OVERRIDES_R2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_topic_labels(filepath: str, max_words: int = 5,
                      round_key: str = "r2") -> dict:
    """
    Returns {topic_num: label_string}.
    Uses round-specific TOPIC_NAME_OVERRIDES when available, otherwise top-5 words.
    round_key: "r1", "r2", or "r3"
    """
    overrides = ROUND_OVERRIDES.get(round_key, TOPIC_NAME_OVERRIDES_R2)

    labels = {}
    path = Path(filepath)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"Topic (\d+): (.+)", line.strip())
            if m:
                num   = int(m.group(1))
                words = ", ".join(m.group(2).split(", ")[:max_words])
                labels[num] = f"T{num:02d}: {words}"

    # Apply human-readable overrides for this round
    for num, name in overrides.items():
        labels[num] = f"T{num:02d}  {name}"

    return labels


def wrap_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(label, width))


def short_docname(name: str, max_len: int = 30) -> str:
    """Shorten filename for axis display."""
    name = Path(name).stem          # strip .txt / .csv
    name = re.sub(r"_+", " ", name) # underscores → spaces
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return name


# ---------------------------------------------------------------------------
# 1. Topic distribution bar chart
# ---------------------------------------------------------------------------
def plot_distribution(chunks_df: pd.DataFrame,
                      topic_labels: dict,
                      round_label: str,
                      out_path: Path) -> None:

    counts = chunks_df["dominant_topic"].value_counts().sort_index()
    n      = len(counts)
    labels = [wrap_label(topic_labels.get(i, f"T{i:02d}")) for i in counts.index]
    pct    = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(13, 0.55 * n + 2))

    bars = ax.barh(range(n), counts.values, color=COLOR_BAR,
                   edgecolor="white", linewidth=0.5, height=0.7)

    # Percentage labels
    for i, (v, p) in enumerate(zip(counts.values, pct)):
        ax.text(v + counts.max() * 0.01, i, f"{p:.1f}%",
                va="center", ha="left", fontsize=8,
                color="#555555")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of chunks")
    ax.set_title(f"Topic distribution — {round_label}\n"
                 f"({counts.sum()} total chunks, {n} topics)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.margins(x=0.12)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# 2. Document–topic heatmap
# ---------------------------------------------------------------------------
def plot_doc_heatmap(chunks_df: pd.DataFrame,
                     topic_labels: dict,
                     round_label: str,
                     out_path: Path,
                     max_docs: int = 50) -> None:

    topic_cols = sorted([c for c in chunks_df.columns
                         if re.match(r"topic_\d+", c)],
                        key=lambda c: int(c.split("_")[1]))

    if not topic_cols:
        print("No topic probability columns found — skipping heatmap.")
        return

    # Average topic proportions per document
    doc_matrix = (chunks_df.groupby("doc_id")[topic_cols]
                  .mean()
                  .round(3))

    # Limit to most topically diverse documents if corpus is large
    if len(doc_matrix) > max_docs:
        diversity = doc_matrix.apply(lambda r: -(r * np.log(r + 1e-9)).sum(),
                                     axis=1)
        doc_matrix = doc_matrix.loc[diversity.nlargest(max_docs).index]

    # Rename axes
    col_labels = [topic_labels.get(int(c.split("_")[1]), c)
                  for c in topic_cols]
    row_labels = [wrap_label(short_docname(d), width=35)
                  for d in doc_matrix.index]

    n_docs   = len(doc_matrix)
    n_topics = len(topic_cols)
    fig_h    = max(8, n_docs * 0.35 + 3)
    fig_w    = max(14, n_topics * 0.9 + 4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        doc_matrix.values,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=doc_matrix.values.max(),
        xticklabels=[wrap_label(l, 20) for l in col_labels],
        yticklabels=row_labels,
        linewidths=0.3,
        linecolor="#eeeeee",
        cbar_kws={"label": "Mean topic proportion", "shrink": 0.6},
    )

    ax.set_title(f"Document–topic distribution — {round_label}\n"
                 f"(darker = stronger topic presence in that document)",
                 pad=14)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Document")
    ax.tick_params(axis="x", rotation=45, labelsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# 3. Topic co-occurrence heatmap
# ---------------------------------------------------------------------------
def plot_cooccurrence(chunks_df: pd.DataFrame,
                      topic_labels: dict,
                      round_label: str,
                      out_path: Path,
                      threshold: float = 0.15) -> None:
    """
    For each document, two topics 'co-occur' if both have mean proportion
    above `threshold`. Counts how many documents share each topic pair.
    """
    topic_cols = sorted([c for c in chunks_df.columns
                         if re.match(r"topic_\d+", c)],
                        key=lambda c: int(c.split("_")[1]))

    if not topic_cols:
        print("No topic columns — skipping co-occurrence.")
        return

    doc_matrix = chunks_df.groupby("doc_id")[topic_cols].mean()
    n          = len(topic_cols)
    cooc       = np.zeros((n, n), dtype=int)

    for _, row in doc_matrix.iterrows():
        active = [i for i, c in enumerate(topic_cols) if row[c] >= threshold]
        for i in active:
            for j in active:
                cooc[i, j] += 1

    # Zero the diagonal (self-co-occurrence is trivial)
    np.fill_diagonal(cooc, 0)

    short_labels = [topic_labels.get(int(c.split("_")[1]), c).split(":")[0]
                    for c in topic_cols]

    fig, ax = plt.subplots(figsize=(11, 9))

    mask = cooc == 0
    sns.heatmap(
        cooc,
        ax=ax,
        mask=mask,
        cmap="Blues",
        annot=True, fmt="d", annot_kws={"size": 8},
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_kws={"label": "Number of documents", "shrink": 0.7},
    )

    ax.set_title(f"Topic co-occurrence across documents — {round_label}\n"
                 f"(cell = no. of documents where both topics are prominent)",
                 pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

ALL_ROUNDS = {
    "r1": {
        "chunks":      BASE / "adenda_r1" / "chunks_topics_r1_k14v2.csv",
        "topics":      BASE / "adenda_r1" / "topics_r1_k14v2.txt",
        "output_dir":  BASE / "adenda_r1",
        "round_label": "Adenda Round 1",
        "round_key":   "r1",
    },
    "r2": {
        "chunks":      BASE / "adenda_r2" / "chunks_topics_r2_k13v1.csv",
        "topics":      BASE / "adenda_r2" / "topics_r2_k13v1.txt",
        "output_dir":  BASE / "adenda_r2",
        "round_label": "Adenda Round 2",
        "round_key":   "r2",
    },
    "r3": {
        "chunks":      BASE / "adenda_r3" / "chunks_topics_r3_k12v2.csv",
        "topics":      BASE / "adenda_r3" / "topics_r3_k12v2.txt",
        "output_dir":  BASE / "adenda_r3",
        "round_label": "Adenda Round 3",
        "round_key":   "r3",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualise LDA topic model outputs — all rounds."
    )
    parser.add_argument("--round", default="all", choices=["r1", "r2", "r3", "all"],
        help="Which round to run (default: all)")
    parser.add_argument("--max_docs", type=int, default=50)
    parser.add_argument("--cooc_threshold", type=float, default=0.15)
    args = parser.parse_args()

    rounds = ALL_ROUNDS if args.round == "all" else {args.round: ALL_ROUNDS[args.round]}

    for rkey, cfg in rounds.items():
        print(f"\n{'='*50}")
        print(f"Processing {cfg['round_label']}...")
        print(f"{'='*50}")

        out = Path(cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

        print(f"Loading {cfg['chunks'].name}...")
        df = pd.read_csv(cfg["chunks"])
        print(f"  {len(df)} chunks, {df['doc_id'].nunique()} documents")

        topic_labels = load_topic_labels(str(cfg["topics"]), round_key=rkey)
        if not topic_labels:
            print("WARNING: no topic labels loaded — using numeric IDs")

        slug = cfg["round_label"].lower().replace(" ", "_")

        plot_distribution(df, topic_labels, cfg["round_label"],
                          out / f"topic_distribution_{slug}.png")
        plot_doc_heatmap(df, topic_labels, cfg["round_label"],
                         out / f"topic_doc_heatmap_{slug}.png",
                         max_docs=args.max_docs)
        plot_cooccurrence(df, topic_labels, cfg["round_label"],
                          out / f"topic_cooccurrence_{slug}.png",
                          threshold=args.cooc_threshold)

    print("\nDone. All rounds visualised.")


if __name__ == "__main__":
    main()