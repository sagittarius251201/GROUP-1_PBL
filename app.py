
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Forecast Fix")

# Sidebar: Data source & page selection
st.sidebar.header("1) Data & Navigation")
source = st.sidebar.radio("Load data via", ["GitHub URL", "Upload CSV"])
if source == "GitHub URL":
    url = st.sidebar.text_input("Raw GitHub CSV URL",
        "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv")
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not uploaded:
        st.sidebar.info("Please upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(uploaded)

page = st.sidebar.selectbox("Select Page", [
    "Visualization",
    "Time Series Forecasting",
    "LTV & Churn",
    "Price Elasticity",
    "Cohort Analysis",
    "Geographic Heatmap",
    "Sentiment Analysis",
    "Competitive Benchmark",
    "ROI Simulator"
])

# Common filters
if "Age" in df.columns:
    age_min, age_max = int(df.Age.min()), int(df.Age.max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
    df = df[df.Age.between(age_range[0], age_range[1])]
if "Gender" in df.columns:
    genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(genders)]
st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

# Page: Time Series Forecasting
if page == "Time Series Forecasting":
    st.header("Time Series Forecasting")
    # Let user select columns
    all_cols = df.columns.tolist()
    date_col = st.selectbox("Select date column", all_cols, index=0)
    value_col = st.selectbox("Select value column", [c for c in all_cols if df[c].dtype in [int, float]], index=0)
    # Parse dates
    df_copy = df[[date_col, value_col]].copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_ts = df_copy.dropna(subset=[date_col, value_col])
    if df_ts.empty:
        st.error("No valid datetime entries found. Please select a correct date column.")
    else:
        df_ts = df_ts.rename(columns={date_col: 'ds', value_col: 'y'})
        m = Prophet()
        m.fit(df_ts)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Forecast shows projected trend for next 30 periods.")

# Other pages would follow similarly...
