# app.py — TruthLens: AI or Authentic?
# Single-file Streamlit app. Put this file and merged_final.csv in the same folder.

import streamlit as st
import pandas as pd
import re
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

# ---------------------------
# Helper functions
# ---------------------------

@st.cache_resource
def load_spacy():
    """Load spaCy model once (cached)."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error(
            "spaCy model 'en_core_web_sm' is not installed. Run:\n"
            "`python -m spacy download en_core_web_sm` in your terminal, then refresh."
        )
        raise

def clean_text(text: str) -> str:
    """Simple text cleaning for TF-IDF. Lowercase, remove non-alphanum, normalize spaces."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tfidf_similarity(a: str, b: str) -> float:
    """Return cosine similarity (0..1) between two strings using TF-IDF."""
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([a, b])
    return float(cosine_similarity(mat[0], mat[1])[0][0])

def entity_overlap(doc1, doc2):
    """Return entities lists and overlap percentage."""
    ents1 = {ent.text for ent in doc1.ents}
    ents2 = {ent.text for ent in doc2.ents}
    common = ents1.intersection(ents2)
    denom = max(len(ents1), len(ents2), 1)
    overlap_pct = round(len(common) / denom * 100, 2)
    return {
        "entities_text1": sorted(list(ents1)),
        "entities_text2": sorted(list(ents2)),
        "common_entities": sorted(list(common)),
        "entity_overlap_pct": overlap_pct
    }

def sentiment_scores(text: str):
    """Return polarity and subjectivity from TextBlob (polarity: -1..1, subjectivity: 0..1)."""
    tb = TextBlob(str(text))
    return {"polarity": round(tb.sentiment.polarity, 3), "subjectivity": round(tb.sentiment.subjectivity, 3)}

def readability_report(text: str):
    """Return a simple readability score dictionary using textstat."""
    try:
        return {
            "Flesch Read Ease": round(textstat.flesch_reading_ease(text), 2),
            "SMOG Index": round(textstat.smog_index(text), 2),
            "Gunning Fog": round(textstat.gunning_fog(text), 2),
            "ARI": round(textstat.automated_readability_index(text), 2),
        }
    except Exception:
        return {"error": "Readability computation failed (text might be too short)."}

# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title="TruthLens: AI or Authentic?", layout="wide")
    st.title("🔎 TruthLens — AI or Authentic?")
    st.caption("Compare two passages (human / AI / other) using similarity, entities, sentiment, and readability.")

    # Load spaCy model
    nlp = load_spacy()

    # Load dataset (merged_final.csv) if present
    st.sidebar.header("Dataset (optional)")
    try:
        df = pd.read_csv("merged_final.csv")
        st.sidebar.write(f"Rows: {len(df)}")
    except FileNotFoundError:
        df = None
        st.sidebar.info("No merged_final.csv found in folder. You can paste texts manually.")

    # Sidebar: quick-fill from dataset
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick-fill from dataset")
    if df is not None:
        if "statement" in df.columns:
            sample_idx = st.sidebar.number_input("Pick row index (0 .. n-1)", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
            if st.sidebar.button("Load sample into Text 1"):
                sample_text = df.loc[int(sample_idx), "statement"]
                st.session_state["text1"] = sample_text
            if st.sidebar.button("Load sample into Text 2"):
                sample_text = df.loc[int(sample_idx), "statement"]
                st.session_state["text2"] = sample_text
        else:
            st.sidebar.write("No 'statement' column found in CSV.")
    else:
        st.sidebar.write("Upload merged_final.csv to enable quick-fill.")

    # Input areas
    st.header("Input texts")
    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("Text 1 (example: human statement)", height=220, key="text1")
    with col2:
        text2 = st.text_area("Text 2 (example: AI response)", height=220, key="text2")

    # Actions
    if st.button("🔬 Analyze texts"):
        if not text1 or not text2:
            st.error("Please provide both Text 1 and Text 2 (paste or use dataset quick-fill).")
        else:
            t1 = text1
            t2 = text2

            st.subheader("1) Cleaned preview")
            c1 = clean_text(t1)
            c2 = clean_text(t2)
            st.write("- Text 1 (cleaned):", c1[:400] + ("..." if len(c1) > 400 else ""))
            st.write("- Text 2 (cleaned):", c2[:400] + ("..." if len(c2) > 400 else ""))

            st.subheader("2) TF-IDF similarity")
            sim = tfidf_similarity(c1, c2)
            st.metric(label="TF-IDF Cosine Similarity", value=f"{sim*100:.2f}%")

            st.subheader("3) Named Entity Overlap (spaCy)")
            doc1 = nlp(t1)
            doc2 = nlp(t2)
            eo = entity_overlap(doc1, doc2)
            st.write(f"Shared entities: {len(eo['common_entities'])} (Overlap: {eo['entity_overlap_pct']}%)")
            st.write("Common entities:", eo["common_entities"])
            with st.expander("Show entities found in Text 1"):
                st.write(eo["entities_text1"])
            with st.expander("Show entities found in Text 2"):
                st.write(eo["entities_text2"])

            st.subheader("4) Sentiment (TextBlob)")
            s1 = sentiment_scores(t1)
            s2 = sentiment_scores(t2)
            st.write("Text 1 sentiment:", s1)
            st.write("Text 2 sentiment:", s2)

            st.subheader("5) Readability (textstat)")
            r1 = readability_report(t1)
            r2 = readability_report(t2)
            st.write("Text 1 readability:", r1)
            st.write("Text 2 readability:", r2)

            st.subheader("Final Verdict")
            if sim >= 0.80:
                st.success("High similarity — likely paraphrase or AI rephrasing.")
            elif sim >= 0.50:
                st.warning("Moderate similarity — some overlap in content or style.")
            else:
                st.info("Low similarity — texts are largely different in wording/content.")

    # Footer
    st.markdown("---")
    st.caption("TruthLens — built for learning. This app compares style & content but does not prove authorship.")

if __name__ == "__main__":
    main()
