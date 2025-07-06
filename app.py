import streamlit as st
import pandas as pd
import numpy as np
# 1) File‐uploader widget (always at top level)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    # 2) Try local file first
    local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
    if local.exists():
        return pd.read_csv(local, parse_dates=["SurveyDate"])
    # 3) Fallback to GitHub raw URL
    url = "https://raw.githubusercontent.com/sagittarius251201/second-/main/health_drink_survey_1000_augmented.csv"
    try:
        df = pd.read_csv(url, parse_dates=["SurveyDate"])
        return df
    except Exception:
        # 4) Finally use uploaded file if present
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file, parse_dates=["SurveyDate"])
        # 5) Otherwise error out
        st.error("No data file found locally or on GitHub, and no upload provided.")
        st.stop()

# Call it once
df = load_data(uploaded)

import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from textblob import TextBlob
import networkx as nx
import time


# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
      .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 16px;
        margin-bottom: 20px;
      }
      abbr { text-decoration: none; }
      abbr[title]:hover { cursor: help; border-bottom: 1px dotted #EF476F; }
    </style>
""", unsafe_allow_html=True)

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Health Drink Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── DATA UPLOAD WIDGET (moved outside cached function) ───────────────────────
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")

# ── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    path = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["SurveyDate"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, parse_dates=["SurveyDate"])
    st.error("No data file found.")
    st.stop()

df = load_data(uploaded)

# ── SIDEBAR FILTERS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")
    age_min, age_max = int(df.Age.min()), int(df.Age.max())
    age = st.slider("Age", age_min, age_max, (age_min, age_max))
    df = df[df.Age.between(*age)]

    gender = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(gender)]

    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]

    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]

    st.download_button("📥 Download CSV", df.to_csv(index=False), "filtered.csv", "text/csv")

# ── TAB NAVIGATION ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "Visualization","Classification","Clustering","Association",
    "Anomaly","Regression","Forecasting","Cohort",
    "Geography","Sentiment","LTV & Churn","Price Elasticity","Glossary"
])

# ── 1. VISUALIZATION ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Quick Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Spend", f"AED {df.SpendPerServing.mean():.2f}")
    c2.metric("Trial Rate", f"{(df.TryNewBrand=='Yes').mean()*100:.1f}%")
    c3.metric("Subscribers", f"{(df.SubscribePlan=='Yes').sum()}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Custom Chart")
    chart = st.selectbox("Choose visualization", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()

    if chart == "Scatter":
        x = st.selectbox("X-axis", nums, key="viz_x")
        y = st.selectbox("Y-axis", nums, key="viz_y")
        fig = px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                         opacity=0.7, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Histogram":
        col = st.selectbox("Feature", nums, key="viz_hist")
        fig = px.histogram(df, x=col, nbins=30, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Box":
        cat = st.selectbox("Category", cats, key="viz_box_cat")
        val = st.selectbox("Value", nums, key="viz_box_val")
        fig = px.box(df, x=cat, y=val, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        cat = st.selectbox("Category", cats, key="viz_bar_cat")
        data = df[cat].value_counts().reset_index()
        data.columns = [cat, "count"]
        fig = px.bar(data, x=cat, y="count", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 2. CLASSIFICATION ────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Classification")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xt)
    Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)

    algo = st.selectbox("Model", ["KNN","Decision Tree","Random Forest","GBRT"])
    if algo == "KNN":
        model = KNeighborsClassifier()
    elif algo == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algo == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()

    model.fit(Xt_s, yt)
    preds = model.predict(Xe_s)
    prob = model.predict_proba(Xe_s)[:,1]

    mets = {
        "Accuracy": accuracy_score(ye, preds),
        "Precision": precision_score(ye, preds),
        "Recall": recall_score(ye, preds),
        "F1": f1_score(ye, preds)
    }
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))

    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig = px.bar(imp, orientation='h', title="Feature Importance", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("What-If Analysis"):
        inp = pd.DataFrame([{
            c: st.slider(c, float(df[c].min()), float(df[c].max()), float(df[c].mean()))
            for c in X.columns
        }])
        p = model.predict(sc.transform(inp))[0]
        st.success("Will Try" if p == 1 else "Will Not Try")

    cm = confusion_matrix(ye, preds)
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred No","Pred Yes"],
        y=["Actual No","Actual Yes"],
        text=cm, texttemplate="%{text}",
        colorscale="Blues"
    ))
    fig_cm.update_layout(title="Confusion Matrix", template="plotly_white")
    st.plotly_chart(fig_cm, use_container_width=True)

    fpr, tpr, _ = roc_curve(ye, prob)
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', line_color="#EF476F"))
    fig_roc.update_layout(title="ROC Curve", template="plotly_white")
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# (continue similarly for tabs[2] through tabs[12], unchanged)

# ── 13. GLOSSARY ────────────────────────────────────────────────────────────
with tabs[12]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📚 Glossary")
    glossary = {
        "SpendPerServing": "AED spent per serving.",
        "HealthConsciousness": "Self-rated health importance (1-10).",
        "TryNewBrand": "Willingness to try a new brand (Yes/No).",
        "SubscribePlan": "Subscription status (Yes/No).",
        "MonthlyDisposableIncome": "Monthly disposable income (AED).",
        "Age": "Age of respondent.",
        "ExerciseFrequency": "How often respondent exercises.",
        "ConsumptionFrequency": "How often health drink is consumed.",
        "Feedback": "Free-text user feedback."
    }
    for k, v in glossary.items():
        st.markdown(f"**{k}** <abbr title='{v}'>ℹ️</abbr>")
    st.markdown('</div>', unsafe_allow_html=True)
