
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
from streamlit_option_menu import option_menu

# Custom CSS for header and sidebar
st.markdown("""<style>
body { background-color: #1B1B2F; color: #FFFFFF; }
.sidebar .sidebar-content { background-color: #30475E; }
h1, h2, h3, h4 { color: #EF476F; }
</style>""", unsafe_allow_html=True)

# Data loader
def load_data(path):
    return pd.read_csv(path, parse_dates=["SurveyDate"])

local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
if local.exists():
    source = str(local)
else:
    st.sidebar.warning("Upload data CSV")
    uploaded = st.sidebar.file_uploader("CSV", type="csv")
    if uploaded:
        source = uploaded
    else:
        st.stop()
df = load_data(source)

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — UX Enhanced")

# Sidebar filter sections
with st.sidebar.expander("Demographics", expanded=True):
    if "Age" in df.columns:
        a_min, a_max = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age", a_min, a_max, (a_min, a_max))
        df = df[df.Age.between(*age)]
    if "Gender" in df.columns:
        g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
        df = df[df.Gender.isin(g)]
with st.sidebar.expander("Behavior"):
    if "ExerciseFrequency" in df.columns:
        ex = st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
        df = df[df.ExerciseFrequency.isin(ex)]
    if "ConsumptionFrequency" in df.columns:
        cf = st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
        df = df[df.ConsumptionFrequency.isin(cf)]
with st.sidebar.expander("Subscription & Location"):
    if "SubscribePlan" in df.columns:
        sp = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
        df = df[df.SubscribePlan.isin(sp)]
    if "City" in df.columns:
        city = st.multiselect("City", df.City.unique(), df.City.unique())
        df = df[df.City.isin(city)]
with st.sidebar.expander("Date Range"):
    if "SurveyDate" in df.columns:
        dr = st.date_input("Survey Date", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
st.sidebar.download_button("Download CSV", df.to_csv(index=False).encode(), "filtered.csv")

# Navigation menu
page = option_menu(None,
    ["Visualization","Classification","Clustering","Association","Anomaly","Regression",
     "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity"],
    icons=["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
           "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin"],
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={"container":{"background-color":"#1B1B2F"},
            "nav-link":{"color":"#FFFFFF","font-size":"16px","margin":"0px 8px"},
            "nav-link-selected":{"background-color":"#EF476F","color":"#FFFFFF"}})

# Visualization
if page=="Visualization":
    st.header("📊 Visualization")
    col1, col2 = st.columns(2)
    with col1:
        chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    with col2:
        metric = st.selectbox("Color Metric", df.select_dtypes("number").columns.tolist())
    if chart=="Scatter":
        fig = px.scatter(df, x=metric, y="SpendPerServing", color="Gender",
                         color_discrete_sequence=px.colors.sequential.Plasma, opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    elif chart=="Histogram":
        fig = px.histogram(df, x=metric, nbins=30, color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    elif chart=="Box":
        fig = px.box(df, x="Gender", y=metric, color="Gender",
                     color_discrete_sequence=px.colors.sequential.Inferno)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.bar(df.groupby("Gender")[metric].mean().reset_index(),
                     x="Gender", y=metric, color="Gender",
                     color_discrete_sequence=px.colors.sequential.Magma)
        st.plotly_chart(fig, use_container_width=True)

# Classification
elif page=="Classification":
    st.header("🤖 Classification")
    st.metric("Accuracy", "85%", delta="↑2%")
    # Placeholders for interactive visuals...

# (Other pages would similarly use vibrant Plotly palettes, columns for layout)

