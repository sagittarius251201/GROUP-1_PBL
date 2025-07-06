
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from textblob import TextBlob
import networkx as nx
import io
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# PDF generation with fallback
def generate_pdf(insights, figs):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Health Drink Market Analysis", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 24))
    for title, text in insights.items():
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 12))
    for idx, (caption, fig) in enumerate(figs.items(), start=1):
        story.append(Paragraph(f"<b>Figure {idx}: {caption}</b>", styles["Heading3"]))
        try:
            img_bytes = fig.to_image(format="png", width=600, height=400)
            img_buf = io.BytesIO(img_bytes)
            story.append(RLImage(img_buf, width=400, height=300))
        except Exception as e:
            story.append(Paragraph(f"*Could not export figure: {e}*", styles["BodyText"]))
        story.append(Spacer(1, 24))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# App config & CSS
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.markdown("""
<style>
.card { background-color: var(--card-bg); padding: 20px; border-radius: 12px;
         box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-bottom: 20px; }
.tooltip { border-bottom: 1px dotted var(--tooltip-fg); cursor: help; }
.light { --card-bg: #fff; --tooltip-fg: #555; }
.dark  { --card-bg: #333; --tooltip-fg: #ddd; }
</style>
""", unsafe_allow_html=True)

# Theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
theme = st.sidebar.radio("Theme", ["Light","Dark"], index=0)
st.session_state.theme = theme
is_dark = (theme == "Dark")
tpl = "plotly_dark" if is_dark else "plotly_white"
st.markdown(f"<body class='{'dark' if is_dark else 'light'}'></body>", unsafe_allow_html=True)

# Load data
def load_data():
    path = Path(__file__).parent/"data"/"health_drink_survey_1000_augmented.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["SurveyDate"])
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        return pd.read_csv(uploaded, parse_dates=["SurveyDate"])
    st.sidebar.error("No data found.")
    st.stop()

if st.sidebar.button("🔄 Refresh Data"):
    df = load_data()
    st.sidebar.success(f"Data refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

# Sidebar filters
filters = []
def add_filter(label, sel):
    if sel:
        text = ", ".join(map(str, sel)) if isinstance(sel, list) else str(sel)
        filters.append(f"{label}: {text}")

with st.sidebar.expander("Demographics", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(g)]; add_filter("Gender", g)
    occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occ)]; add_filter("Occupation", occ)

with st.sidebar.expander("Behavior"):
    ex = st.multiselect("Exercise Frequency", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(ex)]; add_filter("ExerciseFreq", ex)
    cf = st.multiselect("Consumption Frequency", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]; add_filter("ConsumptionFreq", cf)

with st.sidebar.expander("Subscription & Location"):
    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]; add_filter("SubscribePlan", sub)
    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]; add_filter("City", city)

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min().date(), df.SurveyDate.max().date()])
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        df = df[(df.SurveyDate >= start) & (df.SurveyDate <= end)]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("📥 Download Filtered CSV", df.to_csv(index=False).encode(), "filtered.csv")
if filters:
    chips = "  ".join(f"<span class='card tooltip' title='{f}'>{f}</span>" for f in filters)
    st.sidebar.markdown(f"**Active Filters:**  {chips}", unsafe_allow_html=True)

# PDF download button
if st.sidebar.button("📄 Download Analysis (PDF)"):
    insights = {
        "Overall Spend": f"Average spend per serving is AED {df.SpendPerServing.mean():.2f}.",
        "Trial Rate": f"{(df.TryNewBrand=='Yes').mean()*100:.1f}% will try new brand.",
        "Subscribers": f"{(df.SubscribePlan=='Yes').mean()*100:.1f}% are subscribed."
    }
    figs = {
        "Spend vs Income": px.scatter(df, x="MonthlyDisposableIncome", y="SpendPerServing", template=tpl, title="Spend vs Income"),
        "Spend Distribution": px.histogram(df, x="SpendPerServing", nbins=30, template=tpl, title="Spend Distribution")
    }
    pdf_bytes = generate_pdf(insights, figs)
    st.download_button("Save PDF report", pdf_bytes, "health_drink_analysis.pdf", "application/pdf")

# Navigation
pages = ["Visualization","Classification","Clustering","Association","Anomaly","Regression","Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Chat","Glossary"]
icons = ["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up","clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","chat","book"]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal", styles={"nav-link-selected":{"background-color":"#EF476F"}})
st.markdown(f"**Home** → **{page}**")

# 1) Visualization
if page == "Visualization":
    st.header("📊 Visualization")
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='card'><h3>Avg Spend</h3><h2>AED {df.SpendPerServing.mean():.2f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><h3>Trial Rate</h3><h2>{(df.TryNewBrand=='Yes').mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><h3>Subscribers</h3><h2>{(df.SubscribePlan=='Yes').sum()}</h2></div>", unsafe_allow_html=True)
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if chart=="Scatter":
        x=st.selectbox("X-axis",nums,0); y=st.selectbox("Y-axis",nums,1)
        fig=px.scatter(df,x=x,y=y,color=cats[0] if cats else None,opacity=0.7,template=tpl,color_continuous_scale="Turbo")
        st.plotly_chart(fig,use_container_width=True); st.markdown(f"**Insight:** Correlation between **{x}** & **{y}**.")
    elif chart=="Histogram":
        col=st.selectbox("Feature",nums); fig=px.histogram(df,x=col,nbins=30,template=tpl,color_discrete_sequence=["#EF476F"])
        st.plotly_chart(fig,use_container_width=True); st.markdown(f"**Insight:** Mean **{col}** ≈ {df[col].mean():.2f}.")
    elif chart=="Box":
        cat=st.selectbox("Category",cats); val=st.selectbox("Value",nums)
        fig=px.box(df,x=cat,y=val,template=tpl,color_discrete_sequence=["#06D6A0"])
        st.plotly_chart(fig,use_container_width=True); st.markdown(f"**Insight:** Variation of **{val}** across **{cat}**.")
    else:
        cat=st.selectbox("Category",cats); metric=st.selectbox("Metric",["Count","Mean Spend"])
        if metric=="Count":
            data=df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig=px.bar(data,x=cat,y="count",template=tpl,color_discrete_sequence=["#FFD166"])
            st.plotly_chart(fig,use_container_width=True); st.markdown(f"**Insight:** Top segment: **{data.iloc[0][cat]}**.")
        else:
            agg=df.groupby(cat)["SpendPerServing"].mean().reset_index()
            fig=px.bar(agg,x=cat,y="SpendPerServing",template=tpl,color_discrete_sequence=["#EF476F"])
            st.plotly_chart(fig,use_container_width=True)
            top_seg=agg.loc[agg.SpendPerServing.idxmax(),cat]; st.markdown(f"**Insight:** **{top_seg}** spends highest.")

# (Remaining tabs code identical to previously provided full version)
