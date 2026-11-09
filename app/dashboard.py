"""
app/dashboard.py  (FIXED VERSION)
----------------------------------
Fixes:
  1. ValueError on empty sequence → safe_idxmax() guards all idxmax() calls
  2. Live predictor accuracy → enhanced VADER with profanity/slang negative boosting
  3. Filtered data edge cases handled gracefully throughout

Run:
    streamlit run app/dashboard.py
"""


import os
import sys
import pickle
import re
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express       as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict            import predict_ml, load_artifacts
from src.preprocess         import preprocess_dataframe
from src.feature_extraction import load_vectorizer, transform

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}

st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0 0.5rem;
    }
p.bg-icons {
    position: absolute;
    top: 31px;
    font-size: 32px;
    left: 75px;
}

.st-emotion-cache-1bpuen8 p {
    word-break: break-word;
    margin: 0px;
    font-size: 18px !important;
    font-family: "Nunito", sans-serif !important;
    font-weight: 800 !important;
}
.st-emotion-cache-155jwzh.e15d17f12 {
    //background: #00001e !important;
}
.st-emotion-cache-1y4lfdl {
    font-family: "Source Sans", sans-serif;
    font-size: 1.3rem !important;
    margin-bottom: -1rem;
    color: inherit;
    max-width: 100%;
    width: 100%;
    overflow-wrap: break-word;
    font-weight: 400 !important;
}
.st-emotion-cache-11dgbpp p, .st-emotion-cache-6urfhe p {
    word-break: break-word;
    margin: 0px;
    font-size: 16px !important;
    font-family: "Nunito", sans-serif !important;
    font-weight: 600 !important;
}
.st-emotion-cache-16482qj {
    display: flex;
    flex-flow: column wrap;
    align-items: flex-start;
    gap: 10px;
    min-height: 2.5rem;
}
.st-emotion-cache-1wbqy5l.e19wr9s00 {
    visibility: hidden !important;
}
.stToolbarActions.st-emotion-cache-1p1m4ay.epm40z22 {
    visibility: hidden !important;
}
    .sub-header { text-align:center; color:#888; font-size:1rem; margin-bottom:2rem; }
    .stButton>button {
        background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        color:white; border:none; border-radius:8px;
        padding:0.5rem 1.5rem; font-weight:600;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }

.stTabs.st-emotion-cache-8atqhb.e1ac7blb0 {
    margin-bottom: 62px !important;
}

.st-emotion-cache-lhe5w4, .st-emotion-cache-zy6yx3 {
    width: 100%;
    padding: 6rem 1rem 3rem !important;
    max-width: initial;
    min-width: auto;
}

    div[data-testid="stMetricValue"] { font-size:1.8rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Enhanced VADER with slang / profanity / complaint awareness ───────────────

EXTRA_NEGATIVE = {
    "fuck": -3.5, "fucking": -3.5, "fucker": -3.5, "shit": -3.0,
    "shitty": -3.2, "damn": -2.0, "crap": -2.5, "crappy": -2.8,
    "ass": -2.0, "asshole": -3.5, "bastard": -3.0, "bullshit": -3.2,
    "idiot": -2.8, "stupid": -2.5, "dumb": -2.3, "moron": -3.0,
    "useless": -2.8, "pathetic": -3.0, "disgusting": -3.2, "horrible": -3.0,
    "worst": -3.5, "terrible": -3.0, "awful": -3.0, "garbage": -3.2,
    "trash": -3.0, "scam": -3.5, "fraud": -3.5, "liar": -3.2,
    "cheater": -3.2, "ripoff": -3.2, "overpriced": -2.5,
    "disappointed": -2.5, "disappointing": -2.5, "frustrating": -2.8,
    "frustrated": -2.8, "annoying": -2.5, "annoyed": -2.5,
    "angry": -2.8, "furious": -3.2, "hate": -3.0, "hated": -3.0,
    "sucks": -2.8, "suck": -2.8, "sucked": -2.8, "joke": -2.0,
    "nightmare": -3.0, "disaster": -3.0, "ruined": -3.0,
    "broken": -2.5, "damaged": -2.8, "defective": -3.0,
    "fake": -3.0, "counterfeit": -3.2, "late": -1.5, "delayed": -1.8,
    "cancelled": -2.0, "ignored": -2.5, "rude": -2.8,
    "unprofessional": -2.8, "incompetent": -3.0, "waste": -2.5,
    "wasted": -2.5, "unacceptable": -3.0, "ridiculous": -2.5,
    "overcharged": -2.8, "cheated": -3.0, "regret": -2.5,
    "smh": -2.0, "wtf": -2.5, "bruh": -1.0,
}

EXTRA_POSITIVE = {
    "amazing": 3.0, "awesome": 3.0, "fantastic": 3.0, "brilliant": 3.0,
    "superb": 3.0, "outstanding": 3.2, "excellent": 3.0, "perfect": 3.2,
    "loved": 2.8, "love": 2.5, "great": 2.5, "wonderful": 3.0,
    "impressed": 2.5, "satisfied": 2.3, "happy": 2.5, "delighted": 3.0,
    "quick": 1.5, "fast": 1.5, "smooth": 1.8, "easy": 1.5,
    "helpful": 2.5, "polite": 2.3, "friendly": 2.5, "responsive": 2.0,
    "recommend": 2.5, "recommended": 2.5, "worth": 2.0, "affordable": 2.0,
}


def build_enhanced_vader():
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(EXTRA_NEGATIVE)
    sia.lexicon.update(EXTRA_POSITIVE)
    return sia


_SIA = build_enhanced_vader()


def predict_enhanced_vader(texts: list) -> list:
    results = []
    for text in texts:
        cap_ratio  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        cap_boost  = -0.5 if cap_ratio > 0.5 else 0.0
        excl_count = text.count("!")
        excl_boost = -0.1 * min(excl_count, 5)

        scores   = _SIA.polarity_scores(text)
        compound = scores["compound"] + cap_boost + excl_boost
        compound = max(-1.0, min(1.0, compound))

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({
            "text":      text,
            "sentiment": label,
            "compound":  round(compound, 4),
            "pos":       scores["pos"],
            "neu":       scores["neu"],
            "neg":       scores["neg"],
        })
    return results


# ── Safe helpers (FIX for ValueError: argmax of empty sequence) ───────────────

def safe_idxmax(series, default="N/A"):
    """Return idxmax safely — returns default if series is empty or all NaN."""
    try:
        if series.empty or series.isna().all():
            return default
        return series.idxmax()
    except Exception:
        return default


def safe_value_counts_idxmax(df, filter_col, filter_val, group_col, default="N/A"):
    subset = df[df[filter_col] == filter_val][group_col]
    if subset.empty:
        return default
    return safe_idxmax(subset.value_counts(), default)



# ── Sentiment Analysis History (SQLite) ───────────────────────────────────────
# This feature is intentionally kept inside dashboard.py so the existing
# preprocessing/model files do not need to be changed.
HISTORY_DB = os.path.join("data", "sentiment_analysis_history.db")


def init_history_db():
    os.makedirs(os.path.dirname(HISTORY_DB), exist_ok=True)
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT UNIQUE,
                analysis_date TEXT NOT NULL,
                analysis_time TEXT NOT NULL,
                platform TEXT,
                brand TEXT,
                original_text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                score REAL,
                model_used TEXT NOT NULL
            )
        """)
        conn.commit()


def seed_history_from_dataset(df):
    """Load the existing project records once, without duplicating them."""
    init_history_db()
    rows = []
    for _, row in df.iterrows():
        raw_date = pd.to_datetime(row.get("date"), errors="coerce")
        date_str = raw_date.strftime("%Y-%m-%d") if pd.notna(raw_date) else ""
        source_id = f"dataset_{row.get('id', _)}"
        rows.append((
            source_id,
            date_str,
            "00:00:00",
            str(row.get("platform", "")),
            str(row.get("brand", "")),
            str(row.get("text", "")),
            str(row.get("sentiment", "Neutral")),
            None,
            "Existing Dataset",
        ))
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO sentiment_history
            (source_id, analysis_date, analysis_time, platform, brand,
             original_text, sentiment, score, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()


def save_live_results(results, engine, platform="Unknown", brand="Unknown"):
    """Save every Live Predictor result as a new history record."""
    init_history_db()
    now = datetime.now()
    with sqlite3.connect(HISTORY_DB) as conn:
        for i, result in enumerate(results):
            score = result.get("confidence") if engine == "ML Model" else result.get("compound")
            unique_id = f"live_{now.strftime('%Y%m%d%H%M%S%f')}_{i}"
            conn.execute("""
                INSERT INTO sentiment_history
                (source_id, analysis_date, analysis_time, platform, brand,
                 original_text, sentiment, score, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                unique_id,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                platform,
                brand,
                result.get("text", ""),
                result.get("sentiment", "Neutral"),
                score,
                engine,
            ))
        conn.commit()


def load_history():
    init_history_db()
    with sqlite3.connect(HISTORY_DB) as conn:
        return pd.read_sql_query("""
            SELECT id, analysis_date, analysis_time, platform, brand,
                   original_text, sentiment, score, model_used
            FROM sentiment_history
            ORDER BY analysis_date DESC, analysis_time DESC, id DESC
        """, conn)


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/social_media_data.csv"
    if not os.path.exists(path):
        st.error("Dataset not found. Run: `python main.py --generate` first.")
        st.stop()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    seed_history_from_dataset(df)

    vader_res             = predict_enhanced_vader(df["text"].tolist())
    df["vader_sentiment"] = [r["sentiment"] for r in vader_res]
    df["vader_compound"]  = [r["compound"]  for r in vader_res]

    try:
        model, le, vectorizer = load_ml_artifacts_cached()
        if model is not None:
            df2 = preprocess_dataframe(df, text_col="text")
            X   = transform(df2["processed_text"], vectorizer)
            df["ml_sentiment"] = le.inverse_transform(model.predict(X))
        else:
            df["ml_sentiment"] = df["sentiment"]
    except Exception:
        df["ml_sentiment"] = df["sentiment"]

    return df


@st.cache_resource
def load_ml_artifacts_cached():
    try:
        return load_artifacts()
    except Exception:
        return None, None, None


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.title("🎛️ Filters")

    platforms    = ["All"] + sorted(df["platform"].unique().tolist())
    sel_platform = st.sidebar.selectbox("Platform", platforms)

    brands    = ["All"] + sorted(df["brand"].unique().tolist())
    sel_brand = st.sidebar.selectbox("Brand", brands)

    sentiments    = ["All"] + sorted(df["sentiment"].unique().tolist())
    sel_sentiment = st.sidebar.selectbox("True Sentiment", sentiments)

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df["date"].min(), df["date"].max()),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Prediction Engine**")
    engine = st.sidebar.radio("Select", ["ML Model", "VADER (Enhanced)"])
    return sel_platform, sel_brand, sel_sentiment, date_range, engine


def apply_filters(df, platform, brand, sentiment, date_range):
    f = df.copy()
    if platform  != "All": f = f[f["platform"]  == platform]
    if brand     != "All": f = f[f["brand"]     == brand]
    if sentiment != "All": f = f[f["sentiment"] == sentiment]
    try:
        start, end = date_range
        f = f[(f["date"] >= pd.Timestamp(start)) & (f["date"] <= pd.Timestamp(end))]
    except Exception:
        pass
    return f


# ── Tab: Overview ─────────────────────────────────────────────────────────────
def tab_overview(df):
    if df.empty:
        st.warning("No data matches the current filters. Adjust sidebar filters.")
        return

    st.markdown("### 📊 Dataset Overview")
    total  = len(df)
    counts = df["sentiment"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Posts",  f"{total:,}")
    c2.metric("✅ Positive",  f"{counts.get('Positive',0):,}",
              f"{counts.get('Positive',0)/total*100:.1f}%")
    c3.metric("❌ Negative",  f"{counts.get('Negative',0):,}",
              f"{counts.get('Negative',0)/total*100:.1f}%")
    c4.metric("⚪ Neutral",   f"{counts.get('Neutral',0):,}",
              f"{counts.get('Neutral',0)/total*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            title="Sentiment Distribution",
            color=counts.index, color_discrete_map=COLORS, hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=14)
        fig_pie.update_layout(showlegend=True, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        plat = df.groupby(["platform", "sentiment"]).size().reset_index(name="count")
        fig_bar = px.bar(
            plat, x="platform", y="count", color="sentiment",
            title="Posts by Platform & Sentiment",
            color_discrete_map=COLORS, barmode="group", text_auto=True,
        )
        fig_bar.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab: Brand Analysis ───────────────────────────────────────────────────────
def tab_brand_analysis(df):
    if df.empty:
        st.warning("No data matches the current filters.")
        return

    st.markdown("### 🏢 Brand-Level Sentiment Analysis")
    brand_sent = df.groupby(["brand", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(
        brand_sent, x="brand", y="count", color="sentiment",
        title="Sentiment by Brand",
        color_discrete_map=COLORS, barmode="group", text_auto=True,
    )
    fig.update_layout(xaxis_tickangle=-30, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Brand Health Scorecard")
    tot = df.groupby("brand").size()
    pos = df[df["sentiment"] == "Positive"].groupby("brand").size().reindex(tot.index, fill_value=0)
    neg = df[df["sentiment"] == "Negative"].groupby("brand").size().reindex(tot.index, fill_value=0)
    neu = df[df["sentiment"] == "Neutral" ].groupby("brand").size().reindex(tot.index, fill_value=0)

    scorecard = pd.DataFrame({
        "Total Posts":     tot,
        "Positive":        pos,
        "Negative":        neg,
        "Neutral":         neu,
        "Positive Rate %": (pos / tot * 100).round(1),
    }).reset_index().sort_values("Positive Rate %", ascending=False)

    st.dataframe(
        scorecard.style.background_gradient(subset=["Positive Rate %"], cmap="RdYlGn"),
        use_container_width=True,
    )


# ── Tab: Trend Analysis ───────────────────────────────────────────────────────
def tab_trend_analysis(df):
    if df.empty:
        st.warning("No data matches the current filters.")
        return

    st.markdown("### 📈 Sentiment Trend Over Time")
    daily = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig   = px.line(
        daily, x="date", y="count", color="sentiment",
        title="Daily Sentiment Trend",
        color_discrete_map=COLORS, markers=True,
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    if "vader_compound" in df.columns:
        st.markdown("#### 📉 VADER Compound Score Distribution")
        fig2 = px.histogram(
            df, x="vader_compound", color="vader_sentiment",
            nbins=50, title="VADER Compound Score Histogram",
            color_discrete_map=COLORS, opacity=0.75, barmode="overlay",
        )
        fig2.add_vline(x=0.05,  line_dash="dash", line_color="green",
                       annotation_text="Positive threshold")
        fig2.add_vline(x=-0.05, line_dash="dash", line_color="red",
                       annotation_text="Negative threshold")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab: Live Predictor ───────────────────────────────────────────────────────
def tab_live_prediction(engine):
    st.markdown("### 🤖 Live Sentiment Predictor")
    st.markdown(
        "Type any post, complaint, review, or vulgar comment. "
        "**Slang, profanity, ALL CAPS, and repeated !! are all handled correctly.**"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        live_platform = st.selectbox(
            "Platform",
            ["Unknown", "Twitter", "Instagram", "Facebook", "Reddit", "Google Reviews"],
            key="live_platform",
        )
    with col_b:
        live_brand = st.selectbox(
            "Brand",
            ["Unknown", "Apple", "Amazon", "Flipkart", "Netflix", "Samsung",
             "Myntra", "Zomato", "Swiggy", "Uber Eats", "BigBasket"],
            key="live_brand",
        )

    user_text = st.text_area(
        "Enter text",
        placeholder=(
            '"This is absolute garbage!! Never ordering again 😡"\n'
            '"Loved the packaging, delivery was super fast!"\n'
            '"Placed my order. Waiting for delivery."'
        ),
        height=130,
    )

    batch_text = st.text_area(
        "Or paste multiple texts (one per line)",
        placeholder="Post 1\nPost 2\nPost 3",
        height=90,
    )

    if st.button("🔍 Analyze Sentiment"):
        texts = []
        if user_text.strip():
            texts.append(user_text.strip())
        if batch_text.strip():
            texts += [t.strip() for t in batch_text.splitlines() if t.strip()]

        if not texts:
            st.warning("Please enter at least one text.")
            return

        with st.spinner("Analyzing..."):
            if engine == "ML Model":
                model, le, vectorizer = load_ml_artifacts_cached()
                if model is None:
                    st.error("ML model not found. Run `python main.py --train` first.")
                    return
                results     = predict_ml(texts, model, le, vectorizer)
                score_label = "Confidence %"
                score_key   = "confidence"
            else:
                results     = predict_enhanced_vader(texts)
                score_label = "Compound Score"
                score_key   = "compound"

        # New feature: permanently store every live analysis.
        save_live_results(results, engine, live_platform, live_brand)
        st.success(f"Saved {len(results)} analysis record(s) to Sentiment Analysis History.")

        st.markdown("---")
        st.markdown("#### 🎯 Results")
        for r in results:
            sent  = r["sentiment"]
            emoji = {"Positive": "✅", "Negative": "❌", "Neutral": "⚪"}.get(sent, "❓")
            score = r.get(score_key, "N/A")
            color = COLORS.get(sent, "#888")
            st.markdown(
                f"""<div style='background:{color}22; border-left:6px solid {color};
                padding:12px 16px; border-radius:8px; margin-bottom:10px;'>
                <span style='font-size:1.1rem;'><b>{emoji} {sent}</b></span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <span style='color:#555;'>{score_label}: <b>{score}</b></span><br/>
                <span style='color:#333; margin-top:6px; display:block;'>{r['text']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        if len(results) > 1:
            from collections import Counter
            sc  = Counter(r["sentiment"] for r in results)
            fig = px.pie(
                values=list(sc.values()), names=list(sc.keys()),
                color=list(sc.keys()), color_discrete_map=COLORS,
                title="Batch Prediction Summary", hole=0.4,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 How it works"):
        st.markdown("""
| Input | How it's handled |
|---|---|
| Vulgar words (fuck, shit, garbage, trash) | Custom negative lexicon with strong weights |
| ALL CAPS text | Extra negative boost applied |
| Multiple !!! | Urgency penalty added |
| Negations (not good, never again) | VADER natively handles negation context |
| Slang (smh, wtf, bruh) | Added to custom lexicon |
| Positive words (amazing, loved, smooth) | Custom positive lexicon boosts score |

> **Tip:** Use **VADER (Enhanced)** engine for informal/social media text. Use **ML Model** for formal text.
        """)


# ── Tab: Data Explorer ────────────────────────────────────────────────────────
def tab_data_explorer(df):
    if df.empty:
        st.warning("No data matches the current filters.")
        return

    st.markdown("### 🔍 Raw Data Explorer")
    cols = [c for c in ["id", "platform", "brand", "text", "sentiment",
                         "ml_sentiment", "vader_sentiment", "date", "likes"]
            if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, height=400)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "sentiment_data.csv", "text/csv")



# ── Tab: Analysis History ─────────────────────────────────────────────────────
def tab_analysis_history():
    history = load_history()
    st.markdown("### 🗃️ Sentiment Analysis History")
    st.caption("All existing project records plus every new Live Predictor analysis are stored here.")

    if history.empty:
        st.info("No sentiment analysis records are available yet.")
        return

    history["analysis_date"] = pd.to_datetime(history["analysis_date"], errors="coerce")
    history["month"] = history["analysis_date"].dt.strftime("%Y-%m")

    months = sorted(history["month"].dropna().unique().tolist(), reverse=True)
    selected_month = st.selectbox("Select Month", ["All Months"] + months, key="history_month")

    view = history.copy()
    if selected_month != "All Months":
        view = view[view["month"] == selected_month]

    counts = view["sentiment"].value_counts()
    total = len(view)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyses", f"{total:,}")
    c2.metric("Positive", f"{counts.get('Positive', 0):,}")
    c3.metric("Negative", f"{counts.get('Negative', 0):,}")
    c4.metric("Neutral", f"{counts.get('Neutral', 0):,}")

    st.markdown("#### 📊 Sentiment Comparison")
    comparison = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [counts.get("Positive", 0), counts.get("Negative", 0), counts.get("Neutral", 0)],
    })
    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            comparison, x="Sentiment", y="Count", color="Sentiment",
            color_discrete_map=COLORS, text_auto=True,
            title="Positive vs Negative vs Neutral",
        )
        fig_bar.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_pie = px.pie(
            comparison, values="Count", names="Sentiment",
            color="Sentiment", color_discrete_map=COLORS,
            title="Sentiment Percentage", hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 📅 Monthly Sentiment Comparison")
    monthly = (history.groupby(["month", "sentiment"]).size()
               .reset_index(name="count"))
    monthly_pivot = (monthly.pivot(index="month", columns="sentiment", values="count")
                     .fillna(0).reset_index())
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in monthly_pivot.columns:
            monthly_pivot[col] = 0

    fig_month = px.bar(
        monthly_pivot, x="month", y=["Positive", "Negative", "Neutral"],
        barmode="group", color_discrete_map=COLORS,
        title="Monthly Positive, Negative and Neutral Analysis",
        labels={"value": "Number of analyses", "month": "Month", "variable": "Sentiment"},
    )
    fig_month.update_layout(height=420)
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("#### 📈 Monthly Trend")
    fig_line = px.line(
        monthly_pivot, x="month", y=["Positive", "Negative", "Neutral"],
        markers=True, color_discrete_map=COLORS,
        title="Sentiment Trend by Month",
        labels={"value": "Number of analyses", "month": "Month", "variable": "Sentiment"},
    )
    fig_line.update_layout(height=400)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("#### 📋 Stored Analysis Records")
    display = view.copy()
    display["analysis_date"] = display["analysis_date"].dt.strftime("%Y-%m-%d")
    display = display.drop(columns=["month"])
    st.dataframe(display, use_container_width=True, height=450)
    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Analysis History CSV", csv,
        "sentiment_analysis_history.csv", "text/csv",
        key="download_history",
    )

# ── Tab: Insights ─────────────────────────────────────────────────────────────
def tab_insights(df):
    if df.empty:
        st.warning("No data matches the current filters. Adjust sidebar filters.")
        return

    st.markdown("### 💡 Business Insights")
    total   = len(df)
    counts  = df["sentiment"].value_counts()
    pos_pct = counts.get("Positive", 0) / total * 100
    neg_pct = counts.get("Negative", 0) / total * 100

    insights = []

    if pos_pct > 50:
        insights.append(
            f"✅ **Overall positive sentiment is strong ({pos_pct:.1f}%).** "
            f"Marketing can amplify these user posts."
        )

    if neg_pct > 25:
        top_neg = safe_value_counts_idxmax(df, "sentiment", "Negative", "brand")
        insights.append(
            f"⚠️ **High negative volume ({neg_pct:.1f}%).** "
            + (f"*{top_neg}* receives the most complaints — support team should investigate."
               if top_neg != "N/A" else "Review customer feedback urgently.")
        )

    # FIXED: safe guards on every idxmax call
    top_pos_brand = safe_value_counts_idxmax(df, "sentiment", "Positive", "brand")
    if top_pos_brand != "N/A":
        insights.append(
            f"🏆 **{top_pos_brand}** leads in positive mentions. Feature it in campaigns."
        )

    platform_counts = df.groupby("platform").size() if "platform" in df.columns else pd.Series()
    top_platform    = safe_idxmax(platform_counts)
    if top_platform != "N/A":
        insights.append(
            f"📱 **{top_platform}** generates the most posts. Prioritise community management here."
        )

    if not insights:
        st.info("Not enough varied data for insights. Try removing some filters.")
        return

    for ins in insights:
        st.markdown(ins)

    # Brand health chart
    st.markdown("---")
    st.markdown("#### 📊 Brand Health Score")

    tot = df.groupby("brand").size()
    if tot.empty:
        st.info("No brand data for current filters.")
        return

    pos   = df[df["sentiment"] == "Positive"].groupby("brand").size().reindex(tot.index, fill_value=0)
    score = (pos / tot * 100).round(1).reset_index()
    score.columns = ["Brand", "Health Score (%)"]
    score = score.sort_values("Health Score (%)", ascending=True)

    fig = px.bar(
        score, x="Health Score (%)", y="Brand", orientation="h",
        color="Health Score (%)",
        color_continuous_scale=["#F44336", "#FF9800", "#4CAF50"],
        title="Brand Health Score (Positive Rate %)",
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<p class="bg-icons">📊</p><p class="main-header">Social Media Sentiment Analysis Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">🤖 Real-Time AI Intelligence — Understanding Sentiment, One Post at a Time 💬 Powered by NLP & Machine Learning 💡</p>',
        unsafe_allow_html=True,
    )

    df = load_data()

    sel_platform, sel_brand, sel_sentiment, date_range, engine = render_sidebar(df)
    filtered_df = apply_filters(df, sel_platform, sel_brand, sel_sentiment, date_range)

    label = f"Showing **{len(filtered_df):,}** of **{len(df):,}** posts"
    if filtered_df.empty:
        label += " — ⚠️ No results. Adjust filters."
    st.caption(label)

    tabs = st.tabs([
        "📊 Overview",
        "🏢 Brand Analysis",
        "📈 Trends",
        "🤖 Live Predictor",
        "🗃️ Analysis History",
        "🔍 Data Explorer",
        "💡 Insights",
    ])

    with tabs[0]: tab_overview(filtered_df)
    with tabs[1]: tab_brand_analysis(filtered_df)
    with tabs[2]: tab_trend_analysis(filtered_df)
    with tabs[3]: tab_live_prediction(engine)
    with tabs[4]: tab_analysis_history()
    with tabs[5]: tab_data_explorer(filtered_df)
    with tabs[6]: tab_insights(filtered_df)

    st.markdown("---")
    st.markdown(
        "<center><small>✦ Developed 🧠 by <Strong>Purushottam Sah & Vishwraj Singh Rathore </Strong> 🎓 Under the Guidance of <Strong>Prof. Sathwik Chebrolu </small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
