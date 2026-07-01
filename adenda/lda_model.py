"""
lda_model.py
------------
Trains an LDA topic model on the chunks produced by preprocess.py.
Outputs:
  - topics.txt         : top words per topic (human-readable)
  - chunks_topics.csv  : original chunks with dominant topic + distribution
  - topic_doc_matrix.csv : per-document topic proportions (averaged over chunks)
  - pyldavis.html      : interactive visualisation (open in browser)

Usage:
    python lda_model.py --input chunks.csv --n_topics 15

Dependencies:
    pip install pandas scikit-learn nltk pyldavis
    python -m nltk.downloader stopwords
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Stopwords: English + Spanish + domain-specific noise
# ---------------------------------------------------------------------------
def build_stopwords() -> set:
    import nltk
    base = set()
    for lang in ("english", "spanish"):
        try:
            from nltk.corpus import stopwords as sw
            base |= set(sw.words(lang))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords as sw
            base |= set(sw.words(lang))

    # Extra English function/boilerplate words not in NLTK list
    extra_english = {
        "shall", "may", "must", "also", "within", "least",
        "however", "therefore", "thus", "whereas", "said", "per",
        "according", "including", "without", "whether", "upon",
        "figure", "table", "annex", "section", "chapter", "item",
        "addendum", "rev", "page", "source", "elaborated",
        # Document structure boilerplate
        "description", "circumstances", "coordinates", "campaign",
        "location", "update", "supplementary", "complementary",
        "applicable", "carried", "presented", "indicated",
        "requested", "holder", "response", "answer",
    }

    # Project-level terms: describe the whole corpus, not individual topics
    domain_noise = {
        # Spanish remnants from untranslated chunks
        "proyecto", "area", "zona", "fase", "etapa", "medida",
        "componente", "indica", "señala", "considera", "acuerdo",
        "siguiente", "tabla", "figura", "anexo", "seccion",
        "gac", "pangea",
        # English equivalents — whole-corpus terms
        "project", "zone", "stage", "measure", "component",
        "indicates", "notes", "considers", "following", "regarding",
        "related", "established", "mining", "development",
        "production", "concentrate", "rare", "earth", "clay",
        "extraction",
        # NOTE: construction/operation/closure restored —
        # these help distinguish project phase topics
        # Place names and project zone identifiers
        "neptuno", "luna", "maite", "victoria", "norte",
        "concepcion", "biobio", "penco", "lirquen",
        "zav", "essbio", "cemarc", "ree",
        # Atmospheric dispersion model artifacts
        "alxp", "vin", "sglun", "axdpo", "vismin",
        "modeled", "receptor", "station", "nps", "src",
        "ranked", "hour", "average", "concentration",
        # Generic EIA process terms with no topical signal
        "aclara", "adenda", "icsara", "seia", "rseia",
        "titular", "owner", "authority", "service",
        # Geotechnical test jargon / measurement artifacts
        "mai", "friction", "angle", "vibratory", "measurement",
        "polymer", "combustion",
        # Spanish survivors
        "extraccion", "ambiental", "año", "evaluacion",
        "impacto", "solicita", "titular", "presenta",
        # Compliance boilerplate
        "complies", "comply",
        # Emissions/modeling terms causing topic splits
        "receptors", "receptor", "modeled", "observed",
        "daily", "exposure", "model", "analysis",
        # Infrastructure terms that blur topic boundaries
        "roads", "road", "internal", "route", "truck", "vehicle",
        "activities", "local", "municipality",
        # Remaining place name survivors
        "concepcion", "located",
        # Repeated boilerplate phrases
        "year year", "exceed exceed",
        # Vocabulary causing flora/fauna/cultural merge
        "cultural", "access", "influence", "areas",
        "points", "information", "form",
        # Bigram artifacts sklearn generates from adjacent stopwords
        "exceed exceed", "year year", "operation phase",
        "construction phase", "closure phase",
        # Remaining Spanish survivors
        "extraccion", "extracción",
        # Noise measurement station terms still leaking
        "distant", "north", "wild", "birds",
        "story", "observed",
        # Round 1 artifacts: model output noise tokens
        "vnf", "cav",
        # Round 1 artifacts: taxonomic table artifacts
        "insecta", "coleoptera", "liolaemus", "mma",
        # Round 1 artifacts: untranslated Spanish survivors
        "tierras", "arcillas", "raras", "desarrollo",
        # Round 1 artifacts: generic terms with no topical signal
        "value", "study", "used", "high", "low", "total",
        "based", "present", "general", "specific", "given",
        # Round 1 artifacts: archaeological/heritage noise
        "archaeological", "heritage", "view",
        # Exceed artifacts (unigram — bigram already covered above)
        "exceed", "exceeds", "exceeded",
        # Round 1 v3: eucalyptus plantation artifacts
        "eucalyptus", "globulus", "eucalyptus globulus", "plantation",
        # Round 1 v3: place name survivors
        "concepcion", "concepción", "santiago",
        # Round 1 v3: incoherent topic artifacts
        "predator", "nub", "bio", "canal", "seismic",
        # Round 1 v3: overly generic terms
        "level", "could", "results", "report", "site",
        "point", "number", "type", "case", "due",
        # Round 1 v3: legal boilerplate survivors
        "supreme", "decree", "chile",
        # Round 1 v3: laboratory/sampling noise
        "test", "laboratory", "client", "date", "sample",
        # Round 1 k13v1: statistical/cartographic artifacts
        "pca", "cerro", "mau", "record",
        # Round 1 k13v1: generic terms with no topical signal
        "values", "conditions", "levels", "elements",
        # Round 1 k13v1: EIA process boilerplate
        "eia",
        # Round 1 k13v1: office/admin artifacts
        "office",
        # Round 1 k13v2: taxonomic table / coordinate artifacts
        "diptera", "arachnida", "appendix", "utm", "wgs",
        "hymenoptera", "lepidoptera", "orthoptera", "hemiptera",
        "coleoptera", "araneae", "acari", "familia", "orden",
        # Round 1 k13v2: forest/vegetation split — merge signals
        "preservation", "protection",
        # Round 1 k13v2: remaining generic noise
        "material", "design", "machinery", "day",
        # Round 1 k13v3: monitoring station code artifacts
        "ara", "lbo", "lla", "val", "lri", "nub", "bio",
        # Round 1 k13v3: remaining generic noise
        "parameters", "actions", "year", "system",
        "flow", "chemical", "ministry", "community",
        # Round 1 k14v1: measurement/lab code artifacts
        "mstd", "nch", "earths",
        # Round 1 k14v1: channel/contour duplicate artifacts
        "contour", "meters", "considered", "network", "climate",
        "presence", "effects", "external", "control", "compliance",
        # Round 1 k14v1: dissolved/stream noise
        "dissolved", "stream",
        # Round 1 k14v2: lab/consulting firm name artifacts
        "gensis", "engineering", "elevation", "unit",
        # Round 1 k14v2: remaining generic noise
        "use", "environment", "region", "concern",
        "individuals", "human",
        # Round 3 k15v1: taxonomic species artifacts
        "citronella", "mucronata", "lizard",
        # Round 3 k15v1: document code artifacts
        "mps", "conducted", "splp",
        # Round 3 k15v1: generic noise
        "three", "permanent", "contributions", "loss",
        "energy", "storage",
        # Round 3 k13v1: fauna/flora artifacts
        "mouse", "seeds", "size", "distribution",
        # Round 3 k13v1: place name artifacts
        "arauco",
        # Round 3 k13v1: hydrology artifacts
        "iii", "exceptional", "event",
        # Round 3 k12v1: measurement/lab artifacts
        "standard", "standards", "concentrations", "fruits",
        # Round 3 k12v1: translation artifacts
        "yes", "intervened", "biodiversity", "recorded",
        # Round 3 k11v1: remaining artifacts
        "germplasm", "observation", "sites", "activity",
        "equipment", "application", "construction",
        "noise",
        # Round 2 k13v1: generic noise
        "works", "work", "assessment", "along", "washed",
        "annual", "collection", "landscape", "hectares",
        # Round 2 k12v1: restored — phase/operation are meaningful
        # Additional noise only
        "path", "substances", "slope", "habitat", "compensation",
        "significant", "impacts", "article", "specimens",
        "estimated", "stations", "wind", "traffic",
        "mitigation", "monitoring",
        # Round 2 k12v2: plantation artifacts
        "radiata", "pinus", "name",
        # Round 2 k12v2: generic noise
        "proposed", "fire", "law", "environmental",
    }

    return base | extra_english | domain_noise


def tokenize(text: str, stopwords: set, min_len: int = 3) -> list[str]:
    """Lowercase, keep only alphabetic tokens, remove stopwords.
    No stemming — text is English and stemming hurts readability."""
    tokens = re.findall(r"[a-záéíóúüñ]+", text.lower())
    tokens = [
        t for t in tokens
        if len(t) >= min_len and t not in stopwords
    ]
    return tokens


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LDA topic modelling on preprocessed EIA chunks."
    )
    parser.add_argument("--input",     default="chunks.csv",
                        help="CSV produced by preprocess.py")
    parser.add_argument("--text_col",  default="text",
                        help="Column to use as text input (default: text, use text_en for translated)")
    parser.add_argument("--n_topics",  type=int, default=15,
                        help="Number of LDA topics (default: 15)")
    parser.add_argument("--top_words", type=int, default=15,
                        help="Top words to display per topic (default: 15)")
    parser.add_argument("--max_df",    type=float, default=0.85,
                        help="Ignore terms in >X%% of docs (default: 0.85)")
    parser.add_argument("--min_df",    type=int, default=3,
                        help="Ignore terms in <N docs (default: 3)")
    parser.add_argument("--max_iter",  type=int, default=50,
                        help="LDA iterations (default: 50)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_suffix", default="",
                        help="Suffix added to all output filenames, e.g. '_k15' (default: none)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading chunks...")
    df = pd.read_csv(args.input)
    text_col = args.text_col
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"  {len(df)} chunks from {df['doc_id'].nunique()} documents")
    print(f"  Using text column: '{text_col}'")

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    print("Tokenising...")
    stopwords = build_stopwords()
    df["tokens"] = df[text_col].apply(
        lambda t: tokenize(t, stopwords)
    )
    # Join back for sklearn vectoriser
    df["token_str"] = df["tokens"].apply(" ".join)

    # Drop chunks that became empty after tokenisation
    df = df[df["token_str"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"  {len(df)} chunks after tokenisation filter")

    # ------------------------------------------------------------------
    # 3. Vectorise (TF — LDA uses raw counts, not TF-IDF)
    # ------------------------------------------------------------------
    from sklearn.feature_extraction.text import CountVectorizer
    print("Vectorising...")
    vectoriser = CountVectorizer(
        max_df=args.max_df,
        min_df=args.min_df,
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
    )
    dtm = vectoriser.fit_transform(df["token_str"])
    vocab = vectoriser.get_feature_names_out()
    print(f"  Vocabulary: {len(vocab)} terms, DTM shape: {dtm.shape}")

    # ------------------------------------------------------------------
    # 4. Train LDA
    # ------------------------------------------------------------------
    from sklearn.decomposition import LatentDirichletAllocation
    print(f"Training LDA ({args.n_topics} topics, {args.max_iter} iterations)...")
    lda = LatentDirichletAllocation(
        n_components=args.n_topics,
        max_iter=args.max_iter,
        learning_method="batch",    # better for smaller corpora
        random_state=args.random_seed,
        n_jobs=-1,
    )
    doc_topic = lda.fit_transform(dtm)
    print(f"  Done. Log-likelihood: {lda.bound_:.1f}")

    # ------------------------------------------------------------------
    # 5. Save topics.txt
    # ------------------------------------------------------------------
    topic_lines = []
    for i, topic_vec in enumerate(lda.components_):
        top_idx = topic_vec.argsort()[-args.top_words:][::-1]
        top_words = [vocab[j] for j in top_idx]
        line = f"Topic {i:02d}: {', '.join(top_words)}"
        topic_lines.append(line)
        print(line)

    suffix = args.output_suffix
    with open(f"topics{suffix}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(topic_lines))
    print(f"\nSaved topics{suffix}.txt")

    # ------------------------------------------------------------------
    # 6. Annotate chunks with dominant topic + full distribution
    # ------------------------------------------------------------------
    df["dominant_topic"] = doc_topic.argmax(axis=1)
    df["topic_confidence"] = doc_topic.max(axis=1).round(4)
    for i in range(args.n_topics):
        df[f"topic_{i:02d}"] = doc_topic[:, i].round(4)

    df.to_csv(f"chunks_topics{suffix}.csv", index=False, encoding="utf-8")
    print(f"Saved chunks_topics{suffix}.csv")

    # ------------------------------------------------------------------
    # 7. Per-document topic proportions
    # ------------------------------------------------------------------
    topic_cols = [f"topic_{i:02d}" for i in range(args.n_topics)]
    doc_matrix = (
        df.groupby("doc_id")[topic_cols]
        .mean()
        .round(4)
    )
    doc_matrix["dominant_topic"] = doc_matrix[topic_cols].idxmax(axis=1)
    doc_matrix.to_csv(f"topic_doc_matrix{suffix}.csv", encoding="utf-8")
    print(f"Saved topic_doc_matrix{suffix}.csv")

    # ------------------------------------------------------------------
    # 8. pyLDAvis interactive visualisation
    # ------------------------------------------------------------------
    try:
        import pyLDAvis
        import pyLDAvis.lda_model as lda_vis
        print("Generating pyLDAvis visualisation...")
        vis = lda_vis.prepare(lda, dtm, vectoriser, mds="tsne")
        pyLDAvis.save_html(vis, f"pyldavis{suffix}.html")
        print(f"Saved pyldavis{suffix}.html  <- open this in your browser")
    except ImportError:
        print("pyLDAvis not installed — skipping visualisation.")
        print("  Install with: pip install pyldavis")
    except Exception as e:
        print(f"pyLDAvis failed ({e}) — skipping.")

    # ------------------------------------------------------------------
    # 9. Quick summary
    # ------------------------------------------------------------------
    print("\n--- Topic distribution across corpus ---")
    summary = (
        df.groupby("dominant_topic")
        .size()
        .rename("chunk_count")
        .sort_index()
    )
    for topic_id, count in summary.items():
        words = topic_lines[topic_id].split(": ", 1)[1].split(", ")[:5]
        print(f"  Topic {topic_id:02d} ({count:4d} chunks): {', '.join(words)}")


if __name__ == "__main__":
    main()