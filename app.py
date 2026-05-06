import os
import re
import glob
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="PYQ Topic Recommendation System", layout="wide")

st.markdown("""
<style>
.main-title {font-size: 2.3rem; font-weight: 800; color: #1f3b73;}
.sub-title {font-size: 1rem; color: #555; margin-bottom: 1.5rem;}
.metric-card {background: #f6f8ff; padding: 1rem; border-radius: 14px; border: 1px solid #dfe6ff;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">PYQ-Based Important Topic Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NLP web app for PYQ analysis, topic recommendation, similarity detection, syllabus mapping, and exam preparation summary.</div>', unsafe_allow_html=True)

DEFAULT_SYLLABUS = {
    "Programming & Data Structures": "C programming recursion arrays stacks queues linked list trees graphs hashing sorting searching complexity",
    "Algorithms": "greedy dynamic programming divide and conquer graph algorithms shortest path minimum spanning tree complexity np completeness",
    "Operating Systems": "process thread scheduling synchronization semaphore deadlock memory management paging segmentation virtual memory file system",
    "Databases": "er model relational algebra sql normalization transaction concurrency control indexing b tree recovery",
    "Computer Networks": "osi tcp ip routing congestion control transport layer application layer dns http network security",
    "Theory of Computation": "finite automata regular languages context free grammar pushdown automata turing machine decidability",
    "Computer Organization": "cpu pipelining cache memory addressing modes instruction set arithmetic datapath control unit",
    "Digital Logic": "boolean algebra logic gates combinational circuits sequential circuits flip flops counters multiplexers",
    "Compiler Design": "lexical analysis parsing syntax directed translation intermediate code optimization code generation",
    "Mathematics": "discrete mathematics probability linear algebra calculus graph theory combinatorics recurrence relations"
}

@st.cache_resource(show_spinner=False)
def load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def find_col(cols, keys):
    cols_l = {c.lower(): c for c in cols}
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in keys):
            return c
    return None

def load_kaggle_dataset():
    try:
        import kagglehub
        path = kagglehub.dataset_download("sakshi2409/gate-cse-question-classification-dataset")
        files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True) + glob.glob(os.path.join(path, "**", "*.xlsx"), recursive=True)
        if not files:
            return None, "Dataset downloaded, but no CSV/XLSX file was found."
        file = files[0]
        if file.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df, f"Loaded Kaggle dataset from: {os.path.basename(file)}"
    except Exception as e:
        return None, f"Kaggle loading failed: {e}"

def prepare_df(df):
    qcol = find_col(df.columns, ["question", "ques", "text", "problem"])
    tcol = find_col(df.columns, ["topic", "subject", "label", "class", "category"])
    if qcol is None:
        qcol = df.select_dtypes(include="object").columns[0]
    if tcol is None:
        obj_cols = list(df.select_dtypes(include="object").columns)
        tcol = obj_cols[1] if len(obj_cols) > 1 else None
    out = pd.DataFrame()
    out["question"] = df[qcol].astype(str)
    out["clean_question"] = out["question"].apply(clean_text)
    if tcol:
        out["topic"] = df[tcol].astype(str)
    else:
        out["topic"] = "Unknown"
    out = out[out["clean_question"].str.len() > 5].drop_duplicates("clean_question").reset_index(drop=True)
    return out, qcol, tcol

def get_embeddings(texts, mode):
    if mode == "Sentence Transformer / BERT":
        model = load_sentence_model()
        if model is not None:
            return model.encode(texts, show_progress_bar=False), "Sentence Transformer embeddings"
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    emb = vec.fit_transform(texts).toarray()
    return emb, "TF-IDF embeddings fallback"

def parse_syllabus(text):
    mp = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            mp[k.strip()] = v.strip()
    if not mp:
        return DEFAULT_SYLLABUS
    return mp

def syllabus_mapping(df, syllabus, mode):
    qtexts = df["clean_question"].tolist()
    unit_names = list(syllabus.keys())
    unit_texts = [clean_text(k + " " + v) for k, v in syllabus.items()]
    all_texts = qtexts + unit_texts
    emb, used = get_embeddings(all_texts, mode)
    qemb = emb[:len(qtexts)]
    semb = emb[len(qtexts):]
    sims = cosine_similarity(qemb, semb)
    idx = sims.argmax(axis=1)
    score = sims.max(axis=1)
    df2 = df.copy()
    df2["mapped_unit"] = [unit_names[i] for i in idx]
    df2["mapping_score"] = np.round(score, 3)
    return df2, used

def similar_questions(df, query, mode, topn=5):
    texts = [clean_text(query)] + df["clean_question"].tolist()
    emb, used = get_embeddings(texts, mode)
    sims = cosine_similarity([emb[0]], emb[1:])[0]
    ids = np.argsort(sims)[::-1][:topn]
    res = df.iloc[ids][["question", "topic"]].copy()
    res["similarity"] = np.round(sims[ids], 3)
    return res, used

def topic_modeling(df, n_topics=6):
    vec = CountVectorizer(max_features=1200, stop_words="english")
    X = vec.fit_transform(df["clean_question"])
    n_topics = min(n_topics, max(2, len(df)//5))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = np.array(vec.get_feature_names_out())
    rows = []
    for i, comp in enumerate(lda.components_):
        top = words[comp.argsort()[-8:][::-1]]
        rows.append({"Discovered Topic": f"Topic {i+1}", "Top Keywords": ", ".join(top)})
    return pd.DataFrame(rows)

def train_classifier(df):
    if df["topic"].nunique() < 2 or len(df) < 20:
        return None, None
    vc = df["topic"].value_counts()
    valid = vc[vc >= 2].index
    data = df[df["topic"].isin(valid)].copy()
    if data["topic"].nunique() < 2:
        return None, None
    Xtr, Xte, ytr, yte = train_test_split(data["clean_question"], data["topic"], test_size=0.25, random_state=42, stratify=data["topic"])
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    Xtrv = vec.fit_transform(Xtr)
    Xtev = vec.transform(Xte)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtrv, ytr)
    pred = clf.predict(Xtev)
    acc = accuracy_score(yte, pred)
    return (vec, clf), acc

def generate_summary(df):
    top_topics = df["mapped_unit"].value_counts().head(5)
    lines = []
    lines.append("The analysis shows that the most repeated areas are " + ", ".join(top_topics.index.tolist()) + ".")
    lines.append("Students should prioritize topics that have both high frequency and strong syllabus mapping scores.")
    lines.append("Questions with high similarity indicate repeated patterns, so they should be revised first.")
    lines.append("This system does not predict the exact upcoming paper; it recommends high-priority topics based on previous question trends.")
    return " ".join(lines)

with st.sidebar:
    st.header("Input Options")
    source = st.radio("Choose dataset source", ["Upload CSV/Excel", "Use Kaggle Dataset", "Use Sample Dataset"])
    emb_mode = st.selectbox("Embedding method", ["Sentence Transformer / BERT", "TF-IDF"])
    st.caption("BERT may take time on first run because the model is downloaded.")

sample = pd.DataFrame({
    "question": [
        "Explain deadlock prevention and avoidance in operating systems.",
        "What is demand paging? Explain page replacement algorithms.",
        "Discuss normalization and functional dependency in DBMS.",
        "Explain TCP congestion control and flow control.",
        "What are finite automata and regular expressions?",
        "Discuss quick sort and its time complexity.",
        "Explain cache memory mapping techniques.",
        "What is syntax analysis in compiler design?",
        "Explain flip flops and sequential circuits.",
        "Discuss probability and Bayes theorem with example."
    ],
    "topic": ["Operating Systems", "Operating Systems", "Databases", "Computer Networks", "Theory of Computation", "Algorithms", "Computer Organization", "Compiler Design", "Digital Logic", "Mathematics"]
})

raw_df = None
msg = ""
if source == "Upload CSV/Excel":
    up = st.file_uploader("Upload PYQ dataset", type=["csv", "xlsx"])
    if up:
        raw_df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        msg = "Uploaded dataset loaded successfully."
elif source == "Use Kaggle Dataset":
    if st.button("Download and Load Kaggle Dataset"):
        raw_df, msg = load_kaggle_dataset()
else:
    raw_df = sample
    msg = "Sample dataset loaded."

if raw_df is None:
    st.info("Upload a dataset, load the Kaggle dataset, or use the sample dataset from the sidebar.")
    st.stop()

st.success(msg)
df, qcol, tcol = prepare_df(raw_df)

c1, c2, c3 = st.columns(3)
c1.metric("Total Questions", len(df))
c2.metric("Detected Question Column", qcol)
c3.metric("Detected Topic Column", tcol if tcol else "Not found")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Topic Frequency Analysis")
freq = df["topic"].value_counts().reset_index()
freq.columns = ["Topic", "Question Count"]
st.dataframe(freq, use_container_width=True)
st.bar_chart(freq.set_index("Topic"))

st.subheader("Syllabus Mapping")
syllabus_text = st.text_area(
    "Enter syllabus units in Unit: topics format",
    value="\n".join([f"{k}: {v}" for k, v in DEFAULT_SYLLABUS.items()]),
    height=220
)
syllabus = parse_syllabus(syllabus_text)
mapped_df, used = syllabus_mapping(df, syllabus, emb_mode)
st.caption(f"Embedding used: {used}")
st.dataframe(mapped_df[["question", "topic", "mapped_unit", "mapping_score"]].head(20), use_container_width=True)

st.subheader("Important Topic Recommendation")
imp = mapped_df.groupby("mapped_unit").agg(
    question_count=("question", "count"),
    avg_mapping_score=("mapping_score", "mean")
).reset_index()
imp["priority_score"] = np.round(imp["question_count"] * imp["avg_mapping_score"], 3)
imp = imp.sort_values("priority_score", ascending=False)
st.dataframe(imp, use_container_width=True)
st.bar_chart(imp.set_index("mapped_unit")["priority_score"])

st.subheader("Similar Question Detection")
query = st.text_input("Enter a new or previous question", "Explain deadlock and its prevention methods.")
if query:
    sim_df, used2 = similar_questions(df, query, emb_mode, 7)
    st.caption(f"Similarity method used: {used2}")
    st.dataframe(sim_df, use_container_width=True)

st.subheader("Topic Modeling")
n_topics = st.slider("Number of discovered topics", 2, 10, 6)
try:
    topics_df = topic_modeling(df, n_topics)
    st.dataframe(topics_df, use_container_width=True)
except Exception as e:
    st.warning(f"Topic modeling could not run: {e}")

st.subheader("Topic Classification")
model_pack, acc = train_classifier(df)
if model_pack:
    st.write(f"TF-IDF + Logistic Regression classification accuracy: **{acc:.2f}**")
    new_q = st.text_input("Classify a question", "What is virtual memory in operating system?")
    if new_q:
        vec, clf = model_pack
        pred = clf.predict(vec.transform([clean_text(new_q)]))[0]
        st.success(f"Predicted Topic: {pred}")
else:
    st.info("Classification needs at least two topics with enough examples per topic.")

st.subheader("LLM-Style Exam Preparation Summary")
summary = generate_summary(mapped_df)
st.write(summary)

st.subheader("Export Report")
report = mapped_df.copy()
report_csv = report.to_csv(index=False).encode("utf-8")
st.download_button("Download Mapped PYQ Report CSV", report_csv, "pyq_mapped_report.csv", "text/csv")
imp_csv = imp.to_csv(index=False).encode("utf-8")
st.download_button("Download Important Topic Report CSV", imp_csv, "important_topic_report.csv", "text/csv")
