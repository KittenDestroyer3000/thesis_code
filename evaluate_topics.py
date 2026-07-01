"""
evaluate_topics.py
------------------
Computes topic quality metrics for all three final LDA models:

  - C_v coherence   : measures whether top words co-occur in the corpus
                      (0-1, higher is better, >0.5 generally good)
  - NPMI coherence  : normalised pointwise mutual information variant
                      (-1 to 1, higher is better)
  - Topic diversity : proportion of unique words across all topic top-words
                      (0-1, higher = more distinct topics)

Outputs:
  topic_evaluation.csv   — per-topic coherence scores + summary
  topic_evaluation.txt   — human-readable report

Usage:
    python evaluate_topics.py

Dependencies:
    pip install gensim pandas tqdm
    (gensim is separate from sklearn — install if not already present)
"""

import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

MODELS = {
    "r1": {
        "chunks":  BASE / "adenda_r1" / "chunks_r1_translated.csv",
        "topics":  BASE / "adenda_r1" / "topics_r1_k14v2.txt",
        "label":   "Round 1 (k=14, k14v2)",
        "k":       14,
    },
    "r2": {
        "chunks":  BASE / "adenda_r2" / "chunks_r2_translated.csv",
        "topics":  BASE / "adenda_r2" / "topics_r2_k13v1.txt",
        "label":   "Round 2 (k=13, k13v1)",
        "k":       13,
    },
    "r3": {
        "chunks":  BASE / "adenda_r3" / "chunks_r3_translated.csv",
        "topics":  BASE / "adenda_r3" / "topics_r3_k12v2.txt",
        "label":   "Round 3 (k=12, k12v2)",
        "k":       12,
    },
}

OUTPUT_DIR  = BASE / "icsara"
TOP_N_WORDS = 10   # number of top words to use for coherence calculation

# ── Human-readable topic labels ────────────────────────────────────────────────

TOPIC_LABELS = {
    "r1": {
        0: "Native forest & vegetation",
        1: "Flora monitoring & species conservation",
        2: "Drainage & hydrology",
        3: "Water infrastructure & surface works",
        4: "Processing plant & infrastructure",
        5: "Water process & chemical treatment",
        6: "Environmental impact & measures",
        7: "Hazardous waste & emergency planning",
        8: "Project phases (construction/operation/closure)",
        9: "Soil, revegetation & disposal",
        10: "Fauna baseline & sampling",
        11: "Air quality & health risk",
        12: "Fauna relocation & monitoring",
        13: "Indigenous communities (GHPPI)",
    },
    "r2": {
        0: "Native forest & vegetation",
        1: "Water process & chemical treatment",
        2: "Air quality & emissions",
        3: "Project phases & drainage",
        4: "Processing plant & infrastructure",
        5: "Indigenous communities (GHPPI)",
        6: "Fauna relocation & conservation",
        7: "Soil, emergency & drainage",
        8: "Flora monitoring & planting",
        9: "Environmental planning & measures",
        10: "Health risk assessment",
        11: "Hydrology & water characterisation",
        12: "Air quality & health risk",
    },
    "r3": {
        0: "Project phases & environmental impact",
        1: "Flora species & conservation planning",
        2: "Air quality & emissions monitoring",
        3: "Fauna relocation & baseline sampling",
        4: "Water process & chemical treatment",
        5: "Health risk assessment",
        6: "Indigenous communities (GHPPI)",
        7: "Project phases & emissions (operation/closure)",
        8: "Native forest & species conservation",
        9: "Emergency planning & hazardous waste",
        10: "Drainage & hydrology",
        11: "Soil, vegetation & revegetation",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_topic_words(topics_path: Path, top_n: int = TOP_N_WORDS) -> list[list[str]]:
    """Parse topics_kN.txt, return list of top-N word lists per topic."""
    topics = []
    with open(topics_path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"Topic \d+: (.+)", line.strip())
            if m:
                words = [w.strip() for w in m.group(1).split(",")][:top_n]
                # Keep only unigrams for coherence (bigrams cause issues in gensim)
                words = [w for w in words if " " not in w][:top_n]
                topics.append(words)
    return topics


def tokenize_chunks(chunks_path: Path) -> list[list[str]]:
    """Load translated chunks and tokenize to word lists for gensim."""
    print(f"  Loading chunks: {chunks_path.name}")
    df = pd.read_csv(chunks_path)
    df = df.dropna(subset=["text_en"])
    tokens = []
    for text in tqdm(df["text_en"], desc="  Tokenising", leave=False):
        words = re.findall(r"[a-záéíóúüñ]+", str(text).lower())
        words = [w for w in words if len(w) >= 3]
        if words:
            tokens.append(words)
    return tokens


def topic_diversity(topic_words: list[list[str]]) -> float:
    """
    Proportion of unique words across all topics' top words.
    1.0 = all words unique across topics (perfectly diverse)
    0.0 = all topics share the same words (no diversity)
    """
    all_words  = [w for topic in topic_words for w in topic]
    unique     = len(set(all_words))
    total      = len(all_words)
    return round(unique / total, 4) if total else 0.0


def evaluate_model(round_key: str, config: dict) -> dict:
    """Compute coherence and diversity for one model."""
    print(f"\n[{round_key.upper()}] {config['label']}")

    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        print("ERROR: gensim not installed. Run: pip install gensim")
        raise

    # Load data
    topic_words = load_topic_words(config["topics"])
    texts       = tokenize_chunks(config["chunks"])

    print(f"  Topics: {len(topic_words)}, Chunks: {len(texts)}")

    # Build gensim dictionary
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=3, no_above=0.85)

    # Filter topic words to dictionary vocabulary
    topic_words_filtered = [
        [w for w in words if w in dictionary.token2id]
        for words in topic_words
    ]

    # C_v coherence
    print("  Computing C_v coherence...")
    cv_model = CoherenceModel(
        topics=topic_words_filtered,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        topn=TOP_N_WORDS,
    )
    cv_scores     = cv_model.get_coherence_per_topic()
    cv_mean       = cv_model.get_coherence()

    # NPMI coherence
    print("  Computing NPMI coherence...")
    npmi_model = CoherenceModel(
        topics=topic_words_filtered,
        texts=texts,
        dictionary=dictionary,
        coherence="c_npmi",
        topn=TOP_N_WORDS,
    )
    npmi_scores   = npmi_model.get_coherence_per_topic()
    npmi_mean     = npmi_model.get_coherence()

    # Topic diversity
    diversity = topic_diversity(topic_words_filtered)

    # Per-topic results
    labels = TOPIC_LABELS.get(round_key, {})
    per_topic = []
    for i, (cv, npmi) in enumerate(zip(cv_scores, npmi_scores)):
        per_topic.append({
            "round":       round_key,
            "topic_id":    i,
            "topic_label": labels.get(i, f"Topic {i:02d}"),
            "top_words":   ", ".join(topic_words_filtered[i][:8]),
            "cv_score":    round(float(cv), 4),
            "npmi_score":  round(float(npmi), 4),
        })

    print(f"  C_v mean:   {cv_mean:.4f}")
    print(f"  NPMI mean:  {npmi_mean:.4f}")
    print(f"  Diversity:  {diversity:.4f}")

    return {
        "round_key":   round_key,
        "label":       config["label"],
        "k":           config["k"],
        "cv_mean":     round(float(cv_mean), 4),
        "npmi_mean":   round(float(npmi_mean), 4),
        "diversity":   diversity,
        "per_topic":   per_topic,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results  = []
    all_topics   = []
    summary_rows = []

    for round_key, config in MODELS.items():
        result = evaluate_model(round_key, config)
        all_results.append(result)
        all_topics.extend(result["per_topic"])
        summary_rows.append({
            "round":      result["label"],
            "k":          result["k"],
            "cv_mean":    result["cv_mean"],
            "npmi_mean":  result["npmi_mean"],
            "diversity":  result["diversity"],
        })

    # Save per-topic CSV
    topics_df = pd.DataFrame(all_topics)
    csv_path  = OUTPUT_DIR / "topic_evaluation.csv"
    topics_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nPer-topic CSV: {csv_path}")

    # Build text report
    lines = []
    lines.append("=" * 70)
    lines.append("TOPIC MODEL QUALITY EVALUATION")
    lines.append("=" * 70)
    lines.append(f"\nMetrics:")
    lines.append("  C_v coherence  : 0-1, higher is better (>0.5 good)")
    lines.append("  NPMI coherence : -1 to 1, higher is better (>0 acceptable)")
    lines.append("  Diversity      : 0-1, higher = more distinct topics")
    lines.append(f"\n  Top-{TOP_N_WORDS} words used for coherence calculation")

    lines.append("\n\n" + "-" * 70)
    lines.append("SUMMARY BY ROUND")
    lines.append("-" * 70)
    lines.append(f"{'Round':<30} {'k':>4} {'C_v':>8} {'NPMI':>8} {'Diversity':>10}")
    lines.append("-" * 70)
    for row in summary_rows:
        lines.append(f"{row['round']:<30} {row['k']:>4} "
                     f"{row['cv_mean']:>8.4f} {row['npmi_mean']:>8.4f} "
                     f"{row['diversity']:>10.4f}")

    for result in all_results:
        lines.append(f"\n\n{'=' * 70}")
        lines.append(f"{result['label']}")
        lines.append(f"{'=' * 70}")
        lines.append(f"{'Topic':<4} {'C_v':>7} {'NPMI':>7}  Label")
        lines.append("-" * 70)
        for t in result["per_topic"]:
            flag = " ✓" if t["cv_score"] >= 0.5 else ("  " if t["cv_score"] >= 0.4 else " ✗")
            lines.append(f"T{t['topic_id']:02d}  {t['cv_score']:>7.4f} "
                         f"{t['npmi_score']:>7.4f}{flag}  {t['topic_label']}")
        lines.append(f"\n  Top words sample:")
        for t in result["per_topic"]:
            lines.append(f"  T{t['topic_id']:02d}: {t['top_words']}")

    report = "\n".join(lines)
    print("\n" + report)

    report_path = OUTPUT_DIR / "topic_evaluation.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()