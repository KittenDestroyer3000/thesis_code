"""
visualise_coverage.py
---------------------
Produces four publication-ready visualisations from coverage_round_N.csv:

  1. coverage_distribution_{slug}.png     — overall 1/2/3 score breakdown
  2. coverage_by_topic_{slug}.png         — mean coverage score per topic + score stacks
  3. coverage_by_section_{slug}.png       — mean coverage score per ICSARA section
  4. coverage_similarity_{slug}.png       — max_similarity vs coverage_score scatter

Style matches visualise_topics.py and visualise_icsara.py exactly.

Usage:
    python visualise_coverage.py --round r1
    python visualise_coverage.py --round r2
    python visualise_coverage.py --round r3

    python visualise_coverage.py --round r1 ^
        --input  C:\\path\\to\\coverage_round_1.csv ^
        --output_dir C:\\path\\to\\icsara

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
import numpy as np
import pandas as pd
import seaborn as sns

# ── Shared style (mirrors existing scripts) ────────────────────────────────────

FONT_FAMILY  = "DejaVu Sans"
COLOR_1      = "#D55E00"   # vermillion (Okabe-Ito) — not addressed
COLOR_2      = "#F0E442"   # yellow (Okabe-Ito) — partially addressed
COLOR_3      = "#009E73"   # bluish green (Okabe-Ito) — fully addressed
COLOR_MEAN   = "#0072B2"   # blue (Okabe-Ito) — mean score line

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

SCORE_COLORS = {1: COLOR_1, 2: COLOR_2, 3: COLOR_3}
SCORE_LABELS = {1: "Not addressed", 2: "Partially addressed", 3: "Fully addressed"}

# ── Round-specific topic labels ────────────────────────────────────────────────

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

ROUND_LABELS = {"r1": TOPIC_LABELS_R1, "r2": TOPIC_LABELS_R2, "r3": TOPIC_LABELS_R3}

# ── Helpers ────────────────────────────────────────────────────────────────────

def wrap_label(label: str, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(label, width))


def shorten_section(s: str, words: int = 5) -> str:
    s = s.strip()
    parts = s.split(".")
    if len(parts) >= 2:
        roman = parts[0].strip()
        rest  = ".".join(parts[1:]).strip().split()[:words]
        return f"{roman}. {' '.join(rest)}"
    return s[:45]


def section_sort_key(s: str) -> int:
    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
                 "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
                 "XI": 11, "XII": 12, "XIII": 13, "XIV": 14,
                 "XV": 15, "XVI": 16, "XVII": 17, "PREAMBLE": 0}
    prefix = s.split(".")[0].strip()
    return roman_map.get(prefix, 99)


# ── 1. Overall coverage distribution ──────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, round_label: str, out_path: Path) -> None:
    items   = df[df["item_number"].notna()]
    counts  = items["coverage_score"].value_counts().sort_index()
    total   = len(items)
    labels  = [SCORE_LABELS[int(s)] for s in counts.index]
    colors  = [SCORE_COLORS[int(s)] for s in counts.index]
    pcts    = counts / total * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(counts)), counts.values, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.6)

    for i, (v, p) in enumerate(zip(counts.values, pcts)):
        ax.text(v + total * 0.005, i, f"{p:.1f}%  (n={int(v)})",
                va="center", ha="left", fontsize=9, color="#555555")

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of ICSARA items")

    ax.margins(x=0.18)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 2. Coverage by topic ───────────────────────────────────────────────────────

def plot_by_topic(df: pd.DataFrame, topic_labels: dict,
                  round_label: str, out_path: Path) -> None:
    items      = df[df["item_number"].notna()].copy()
    topic_order = sorted(items["topic_code"].unique())
    labels     = [wrap_label(topic_labels.get(t, t)) for t in topic_order]
    n          = len(topic_order)

    # Stacked bar: proportion of 1/2/3 per topic
    stacked = pd.crosstab(items["topic_code"], items["coverage_score"],
                          normalize="index") * 100
    stacked = stacked.reindex(topic_order)

    # Per-topic stats for error bars
    topic_counts = items.groupby("topic_code")["coverage_score"].count().reindex(topic_order)
    means        = items.groupby("topic_code")["coverage_score"].mean().reindex(topic_order)

    # 95% bootstrapped CI (2000 resamples, appropriate for ordinal data)
    rng = np.random.default_rng(42)
    ci95_lower = pd.Series(index=topic_order, dtype=float)
    ci95_upper = pd.Series(index=topic_order, dtype=float)
    for t in topic_order:
        scores = items[items["topic_code"] == t]["coverage_score"].values
        if len(scores) < 2:
            ci95_lower[t] = means[t]
            ci95_upper[t] = means[t]
            continue
        boot_means = np.array([
            rng.choice(scores, size=len(scores), replace=True).mean()
            for _ in range(2000)
        ])
        ci95_lower[t] = np.percentile(boot_means, 2.5)
        ci95_upper[t] = np.percentile(boot_means, 97.5)
    ci95 = (means - ci95_lower + ci95_upper - means) / 2  # symmetric approx for errorbar

    fig, ax = plt.subplots(figsize=(13, 0.6 * n + 2))

    left = np.zeros(n)
    for score in [1, 2, 3]:
        if score in stacked.columns:
            vals = stacked[score].fillna(0).values
            ax.barh(range(n), vals, left=left,
                    color=SCORE_COLORS[score], label=SCORE_LABELS[score],
                    edgecolor="white", linewidth=0.3, height=0.7)
            left += vals

    # Sample size annotations at end of each bar
    for i, (t, cnt) in enumerate(zip(topic_order, topic_counts)):
        ax.text(101, i, f"n={int(cnt)}", va="center", ha="left",
                fontsize=7.5, color="#555555")

    # Mean score + 95% CI as scatter overlay — normalised to percentage scale
    means_pct = (means.values - 1) / 2 * 100
    ci95_pct  = ci95.values / 2 * 100
    ax.scatter(means_pct, range(n), color=COLOR_MEAN, zorder=5,
               s=40, marker="D")
    ci95_lower_pct = (means.values - ci95_lower.values) / 2 * 100
    ci95_upper_pct = (ci95_upper.values - means.values) / 2 * 100
    ax.errorbar(means_pct, range(n),
                xerr=[ci95_lower_pct, ci95_upper_pct],
                fmt="none", color=COLOR_MEAN, capsize=3,
                linewidth=1.0, zorder=4)
    ax2 = ax.twiny()
    ax2.set_xlim(0, 115)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels(["1.0", "1.5", "2.0", "2.5", "3.0"],
                        color=COLOR_MEAN, fontsize=8)
    # Label moved to figure caption
    ax2.set_xlabel("")
    ax2.spines["top"].set_color(COLOR_MEAN)
    ax2.tick_params(axis="x", colors=COLOR_MEAN, labelsize=8)
    ax2.spines["top"].set_color(COLOR_MEAN)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of items in topic")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlim(0, 115)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
              fontsize=8, framealpha=0, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 3. Coverage by section ─────────────────────────────────────────────────────

def plot_by_section(df: pd.DataFrame, round_label: str, out_path: Path) -> None:
    items = df[df["item_number"].notna()].copy()
    items["section_short"] = items["section"].apply(shorten_section)

    section_order = sorted(items["section_short"].unique(),
                           key=lambda s: section_sort_key(s))
    n = len(section_order)

    stacked = pd.crosstab(items["section_short"], items["coverage_score"],
                          normalize="index") * 100
    stacked = stacked.reindex(section_order)

    fig, ax = plt.subplots(figsize=(13, 0.65 * n + 2))

    left = np.zeros(n)
    for score in [1, 2, 3]:
        if score in stacked.columns:
            vals = stacked[score].fillna(0).values
            ax.barh(range(n), vals, left=left,
                    color=SCORE_COLORS[score], label=SCORE_LABELS[score],
                    edgecolor="white", linewidth=0.3, height=0.7)
            left += vals

    # Mean score overlay — normalised to percentage scale (score 1-3 → 0-100%)
    means = items.groupby("section_short")["coverage_score"].mean().reindex(section_order)
    means_pct = (means.values - 1) / 2 * 100  # map 1-3 → 0-100
    ax.scatter(means_pct, range(n), color=COLOR_MEAN, zorder=5,
               s=40, marker="D")
    ax2 = ax.twiny()
    ax2.set_xlim(0, 122)  # match main axis
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels(["1.0", "1.5", "2.0", "2.5", "3.0"],
                        color=COLOR_MEAN, fontsize=8)
    # Label moved to figure caption
    ax2.set_xlabel("")
    ax2.spines["top"].set_color(COLOR_MEAN)
    ax2.tick_params(axis="x", colors=COLOR_MEAN, labelsize=8)

    ax.set_yticks(range(n))
    ax.set_yticklabels(section_order, fontsize=8.5)
    ax.set_xlabel("% of items in section")
    # Add n= annotations
    sec_counts = items.groupby("section_short")["coverage_score"].count().reindex(section_order)
    for i, cnt in enumerate(sec_counts):
        label = f"n={int(cnt)}"
        if int(cnt) <= 2:
            label += " *"
        ax.text(101, i, label, va="center", ha="left",
                fontsize=7.5, color="#555555")
    ax.set_xlim(0, 122)


    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
              fontsize=8, framealpha=0, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── 4. Similarity vs coverage scatter ─────────────────────────────────────────

def plot_similarity_scatter(df: pd.DataFrame, round_label: str,
                            out_path: Path) -> None:
    items = df[df["item_number"].notna()].copy()
    items["coverage_score"] = items["coverage_score"].astype(int)
    items["max_similarity"]  = items["max_similarity"].astype(float)

    # Jitter coverage score for visibility
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(items))
    y      = items["coverage_score"].values + jitter

    colors = [SCORE_COLORS[s] for s in items["coverage_score"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(items["max_similarity"], y, c=colors, alpha=0.5,
               s=18, edgecolors="none")

    # Mean similarity per score
    means = items.groupby("coverage_score")["max_similarity"].mean()
    for score, mean in means.items():
        ax.axvline(mean, color=SCORE_COLORS[int(score)],
                   linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(mean + 0.003, int(score) + 0.25, f"{mean:.3f}",
                color=SCORE_COLORS[int(score)], fontsize=8)

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([SCORE_LABELS[s] for s in [1, 2, 3]])
    ax.set_xlabel("Max cosine similarity (best matching chunk)")

    ax.set_xlim(items["max_similarity"].min() - 0.02,
                items["max_similarity"].max() + 0.02)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SCORE_COLORS[s], label=SCORE_LABELS[s])
                       for s in [1, 2, 3]]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

ALL_ROUNDS = {
    "r1": {
        "input":       BASE / "icsara" / "data" / "coverage_round_1.csv",
        "output_dir":  BASE / "icsara",
        "round_label": "ICSARA Round 1",
        "round_key":   "r1",
    },
    "r2": {
        "input":       BASE / "icsara" / "data" / "coverage_round_2.csv",
        "output_dir":  BASE / "icsara",
        "round_label": "ICSARA Round 2",
        "round_key":   "r2",
    },
    "r3": {
        "input":       BASE / "icsara" / "data" / "coverage_round_3.csv",
        "output_dir":  BASE / "icsara",
        "round_label": "ICSARA Round 3",
        "round_key":   "r3",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualise ICSARA coverage scoring results — all rounds."
    )
    parser.add_argument("--round", default="all", choices=["r1", "r2", "r3", "all"],
        help="Which round to run (default: all)")
    args = parser.parse_args()

    rounds = ALL_ROUNDS if args.round == "all" else {args.round: ALL_ROUNDS[args.round]}

    for rkey, cfg in rounds.items():
        print(f"\n{'='*50}")
        print(f"Processing {cfg['round_label']}...")

        out         = Path(cfg["output_dir"])
        topic_labels = ROUND_LABELS.get(rkey, TOPIC_LABELS_R2)
        out.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(cfg["input"])
        print(f"  {len(df[df['item_number'].notna()])} scored items")

        slug = cfg["round_label"].lower().replace(" ", "_")

        plot_distribution(df, cfg["round_label"],
                          out / f"coverage_distribution_{slug}.png")
        plot_by_topic(df, topic_labels, cfg["round_label"],
                      out / f"coverage_by_topic_{slug}.png")
        plot_by_section(df, cfg["round_label"],
                        out / f"coverage_by_section_{slug}.png")
        plot_similarity_scatter(df, cfg["round_label"],
                                out / f"coverage_similarity_{slug}.png")

    print("\nDone. All rounds visualised.")


if __name__ == "__main__":
    main()