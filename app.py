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

# Data loading
def load_data(path_or_buf):
    df = pd.read_csv(path_or_buf, parse_dates=["SurveyDate"])
    return df

local_path = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
if local_path.exists():
    data_source = str(local_path)
else:
    st.sidebar.warning("❗️Augmented data not found in 'data/'. Please upload CSV.")
    upload = st.sidebar.file_uploader("Upload augmented CSV", type="csv", key="data_upload")
    if upload:
        data_source = upload
    else:
        st.sidebar.stop()

df = load_data(data_source)

# App config
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Integrated Ultimate")

# Sidebar filters
st.sidebar.header("Filters")
if "Age" in df.columns:
    amin, amax = int(df.Age.min()), int(df.Age.max())
    age_range = st.sidebar.slider("Age range", amin, amax, (amin, amax))
    df = df[df.Age.between(*age_range)]
if "Gender" in df.columns:
    genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(genders)]
if "Occupation" in df.columns:
    occs = st.sidebar.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occs)]
if "ExerciseFrequency" in df.columns:
    exf = st.sidebar.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(exf)]
if "ConsumptionFrequency" in df.columns:
    cf = st.sidebar.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]
if "SubscribePlan" in df.columns:
    subs = st.sidebar.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(subs)]
if "City" in df.columns:
    cities = st.sidebar.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(cities)]
if "SurveyDate" in df.columns:
    drange = st.sidebar.date_input("Survey Date range",
                                   [df.SurveyDate.min().date(), df.SurveyDate.max().date()])
    df = df[df.SurveyDate.between(pd.to_datetime(drange[0]), pd.to_datetime(drange[1]))]

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

# Top navigation
selected = option_menu(
    menu_title=None,
    options=[
        "Visualization","Classification","Clustering","Association",
        "Anomaly","Regression","Forecasting","Cohort",
        "Geography","Sentiment","LTV & Churn","Price Elasticity"
    ],
    icons=[
        "bar-chart","cpu","people","diagram-3","exclamation-triangle",
        "graph-up","clock-history","calendar","geo","chat-dots",
        "cash-stack","cash-coin"
    ],
    menu_icon="grid-3x3-gap",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding":"0!important","background-color":"#F0F3FB"},
        "icon": {"color":"#026EBD","font-size":"18px"},
        "nav-link": {"font-size":"16px","text-align":"center","margin":"0px 8px","--hover-color":"#E0E8F9"},
        "nav-link-selected": {"background-color":"#026EBD","color":"white"},
    }
)

# 1️⃣ Visualization
if selected == "Visualization":
    st.header("Visualization")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if chart == "Scatter" and len(num_cols) >= 2:
        x = st.selectbox("X-axis", num_cols, 0)
        y = st.selectbox("Y-axis", num_cols, 1)
        color = st.selectbox("Color by (optional)", [None]+cat_cols, 0)
        fig = px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Correlation between `{x}` & `{y}`: **{df[x].corr(df[y]):.2f}**.")
    elif chart == "Histogram" and num_cols:
        col = st.selectbox("Feature", num_cols, 0)
        bins = st.slider("Bins", 10, 50, 30)
        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Mode of `{col}`: **{df[col].mode().iloc[0]}**.")
    elif chart == "Box" and num_cols and cat_cols:
        cat = st.selectbox("Category", cat_cols, 0)
        val = st.selectbox("Value", num_cols, 0)
        fig = px.box(df, x=cat, y=val, title=f"{val} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Median `{val}` by `{cat}`: {df.groupby(cat)[val].median().to_dict()}.")
    elif chart == "Bar" and cat_cols:
        cat = st.selectbox("Category", cat_cols, 0)
        metric = st.selectbox("Metric", ["Count","Mean Spend","Sum Spend"], 0)
        if metric == "Count":
            data = df[cat].value_counts().reset_index()
            data.columns = [cat,"count"]
            fig = px.bar(data, x=cat, y="count", title=f"Count by {cat}")
        else:
            agg = df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", title=f"{metric} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** `{metric}` across `{cat}`.")

# 2️⃣ Classification
elif selected == "Classification":
    st.header("Classification")
    X = df.select_dtypes(include="number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (
        KNeighborsClassifier() if algo=="KNN" else
        DecisionTreeClassifier() if algo=="Decision Tree" else
        RandomForestClassifier() if algo=="Random Forest" else
        GradientBoostingClassifier()
    )
    model.fit(Xtr_s, ytr)
    yp = model.predict(Xte_s)
    yp_prob = model.predict_proba(Xte_s)[:,1]
    st.json({
        "Accuracy": accuracy_score(yte,yp),
        "Precision": precision_score(yte,yp),
        "Recall": recall_score(yte,yp),
        "F1": f1_score(yte,yp)
    })
    cm = confusion_matrix(yte,yp)
    fig = go.Figure(data=go.Heatmap(z=cm, x=["No","Yes"], y=["No","Yes"], colorscale="Viridis"))
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    fpr, tpr, _ = roc_curve(yte, yp_prob)
    roc_fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode="lines"))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(roc_fig, use_container_width=True)
    st.markdown("**Insight:** Metrics guide targeted marketing strategies.")

# 3️⃣ Clustering
elif selected == "Clustering":
    st.header("Clustering")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    k = st.slider("Clusters (k)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(df[feats])
    df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig = px.bar(centers, x=centers.index, y=feats, barmode="group", title="Cluster Centers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Identify distinct personas for personalized campaigns.")

# ... and so on for the remaining tabs as in the code above
