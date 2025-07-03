
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

# Style
st.markdown("""<style>
body { background-color: #1B1B2F; color: #FFFFFF; }
.sidebar .sidebar-content { background-color: #30475E; }
h1, h2, h3, h4 { color: #EF476F; }
</style>""", unsafe_allow_html=True)

# Load data
def load_data(path):
    return pd.read_csv(path, parse_dates=["SurveyDate"])
local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
if local.exists():
    df = load_data(local)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = load_data(uploaded)
    else:
        st.stop()

# Page config
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — UI/UX Tabs")

# Sidebar filters
with st.sidebar.expander("Demographics", expanded=True):
    if "Age" in df.columns:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
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
with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df.columns:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
st.sidebar.download_button("Download CSV", df.to_csv(index=False).encode(), "filtered.csv")

# Navigation
page = option_menu(None,
    ["Visualization","Classification","Clustering","Association","Anomaly","Regression",
     "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity"],
    icons=["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
           "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin"],
    orientation="horizontal", menu_icon="cast", default_index=0,
    styles={"container":{"background-color":"#1B1B2F"},
            "nav-link":{"color":"#FFFFFF","font-size":"16px","margin":"0px 8px"},
            "nav-link-selected":{"background-color":"#EF476F","color":"#FFFFFF"}}
)

# Visualization
if page == "Visualization":
    st.header("📊 Visualization")
    chart = st.selectbox("Chart", ["Scatter","Histogram","Box","Bar"])
    num = df.select_dtypes("number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    if chart == "Scatter":
        x = st.selectbox("X", num, 0); y = st.selectbox("Y", num, 1)
        fig = px.scatter(df, x=x, y=y, color=cat[0] if cat else None,
                         color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
    # ... other charts similarly ...

# Classification
elif page == "Classification":
    st.header("🤖 Classification")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler=StandardScaler().fit(Xtr); Xtr_s,Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    algo = st.selectbox("Algo", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (KNeighborsClassifier() if algo=="KNN" else
             DecisionTreeClassifier() if algo=="Decision Tree" else
             RandomForestClassifier() if algo=="Random Forest" else
             GradientBoostingClassifier())
    model.fit(Xtr_s,ytr); pred=model.predict(Xte_s); prob=model.predict_proba(Xte_s)[:,1]
    st.write(f"Accuracy: {accuracy_score(yte,pred):.2f}")
    cm = confusion_matrix(yte,pred)
    fig = go.Figure(go.Heatmap(z=cm, x=["No","Yes"], y=["No","Yes"], colorscale="Plasma"))
    st.plotly_chart(fig, use_container_width=True)

# ... implement all remaining pages similarly ...
