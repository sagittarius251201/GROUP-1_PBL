
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
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
import time

# --- Session State & Config ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")

# --- Theme Switcher ---
theme = st.sidebar.radio("Theme", ['Light','Dark'], index=0)
st.session_state.theme = theme
if theme == 'Dark':
    st.markdown("""<style>
        body {background-color:#1B1B2F; color:#FFFFFF;}
        .sidebar .sidebar-content {background-color:#30475E;}
        h1,h2,h3,h4 {color:#EF476F;}
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        body {background-color:#FFFFFF; color:#000000;}
        .sidebar .sidebar-content {background-color:#F8F9FB;}
        h1,h2,h3,h4 {color:#EF476F;}
    </style>""", unsafe_allow_html=True)

# --- Logo Upload ---
logo = st.sidebar.file_uploader("Upload Logo", type=["png","jpg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

# --- Data Loading & Refresh ---
def load_data():
    local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
    if local.exists():
        return pd.read_csv(local, parse_dates=["SurveyDate"])
    upload = st.sidebar.file_uploader("Upload data CSV", type="csv", key="data")
    if upload:
        return pd.read_csv(upload, parse_dates=["SurveyDate"])
    st.sidebar.error("Data file not found.")
    st.stop()

if st.sidebar.button("Refresh Data"):
    df = load_data()
    st.sidebar.write(f"Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

# --- Sidebar Filters & Chips ---
filters = []
with st.sidebar.expander("Demographics", expanded=True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    if "Gender" in df:
        g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
        df = df[df.Gender.isin(g)]
        filters.append("Gender: " + ", ".join(g))
    if "Occupation" in df:
        occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
        df = df[df.Occupation.isin(occ)]
        filters.append("Occ: " + ", ".join(occ))
with st.sidebar.expander("Behavior"):
    if "ExerciseFrequency" in df:
        ex = st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
        df = df[df.ExerciseFrequency.isin(ex)]
        filters.append("ExFreq: " + ", ".join(ex))
    if "ConsumptionFrequency" in df:
        cf = st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
        df = df[df.ConsumptionFrequency.isin(cf)]
        filters.append("ConFreq: " + ", ".join(cf))
with st.sidebar.expander("Subscription & Location"):
    if "SubscribePlan" in df:
        sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
        df = df[df.SubscribePlan.isin(sub)]
        filters.append("SubPlan: " + ", ".join(sub))
    if "City" in df:
        city = st.multiselect("City", df.City.unique(), df.City.unique())
        df = df[df.City.isin(city)]
        filters.append("City: " + ", ".join(city))
with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

if filters:
    chips = " ".join([f"<span style='background:#EF476F;color:#FFF;padding:4px 8px;border-radius:4px;margin:2px'>{f}</span>" for f in filters])
    st.markdown("**Active Filters:** " + chips, unsafe_allow_html=True)

# --- Breadcrumbs & Navigation ---
pages = ["Visualization","Classification","Clustering","Association","Anomaly","Regression","Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Changelog"]
icons = ["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up","clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","file-text"]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal", styles={"container":{"padding":"0px"},"nav-link":{"font-size":"14px"},"nav-link-selected":{"background-color":"#EF476F"}})
st.markdown(f"**Home** > **{page}**")

# Pages implementation follows previous fully featured code...
# (Ensure each tab block matches functionality from the last integrated version)

# Visualization
if page=="Visualization":
    # ... code from earlier ...

# Classification
elif page=="Classification":
    # ... code ...

# Clustering
elif page=="Clustering":
    # ... code ...

# Association
elif page=="Association":
    # ... code ...

# Anomaly
elif page=="Anomaly":
    # ... code ...

# Regression
elif page=="Regression":
    # ... code ...

# Forecasting
elif page=="Forecasting":
    # ... code ...

# Cohort
elif page=="Cohort":
    # ... code ...

# Geography
elif page=="Geography":
    # ... code ...

# Sentiment
elif page=="Sentiment":
    # ... code ...

# LTV & Churn
elif page=="LTV & Churn":
    # ... code ...

# Price Elasticity
elif page=="Price Elasticity":
    # ... code ...

# Changelog
elif page=="Changelog":
    changelog = Path(__file__).parent / "CHANGELOG.md"
    if changelog.exists():
        st.markdown(changelog.read_text())
    else:
        st.info("No changelog found.")

# Feedback
if st.button("💡 Send Feedback"):
    fb = st.text_area("Your feedback:")
    if st.button("Submit"):
        with open("feedback.txt","a") as f:
            f.write(fb+"\n---\n")
        st.success("Thank you!")
