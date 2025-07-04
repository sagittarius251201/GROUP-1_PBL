
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Health Drink Dashboard", layout="wide", page_icon="🥤")
st.title("🥤 Health Drink Dashboard - UAE Launch Insights")

@st.cache_data
def load_data():
    df = pd.read_csv("health_drink_survey_1000_augmented.csv", encoding="utf-8")
    return df

df0 = load_data()

# Sidebar - Smart Filters
with st.sidebar:
    st.header("🔎 Filters")
    city = st.multiselect("City", sorted(df0["City"].unique()), default=sorted(df0["City"].unique()))
    gender = st.multiselect("Gender", df0["Gender"].unique(), default=list(df0["Gender"].unique()))
    age = st.slider("Age", int(df0["Age"].min()), int(df0["Age"].max()), (int(df0["Age"].min()), int(df0["Age"].max())))
    inc = st.slider("Monthly Disposable Income (AED)", int(df0["MonthlyDisposableIncome"].min()), int(df0["MonthlyDisposableIncome"].max()),
                    (int(df0["MonthlyDisposableIncome"].min()), int(df0["MonthlyDisposableIncome"].max())))
    st.markdown("---")
    page = st.radio(
        "Navigate", [
            "Visualization", "Classification", "Clustering", "Regression", "Association",
            "Anomaly", "Forecasting", "Cohort", "Geography", "Sentiment", "LTV & Churn", "Price Elasticity"
        ],
        index=0
    )

def filter_df(df):
    df = df[
        df["City"].isin(city) &
        df["Gender"].isin(gender) &
        df["MonthlyDisposableIncome"].between(*inc) &
        df["Age"].between(*age)
    ]
    return df

df = filter_df(df0.copy())
if df.empty:
    st.warning("No data for selected filters. Please broaden your selection.")
    st.stop()

def safe_col(df, col):
    return col if col in df.columns else None

# Visualization Tab
if page == "Visualization":
    st.header("📊 Data Visualization")
    nums = df.select_dtypes(include=np.number).columns.tolist()
    cats = df.select_dtypes(include="object").columns.tolist()

    chart = st.selectbox("Chart Type", ["Scatter", "Histogram", "Box", "Bar", "Pie", "Violin", "Line", "Stacked Bar"])
    if chart == "Scatter":
        x = st.selectbox("X-axis", nums)
        y = st.selectbox("Y-axis", nums)
        color = st.selectbox("Color By", ["None"] + cats)
        fig = px.scatter(df, x=x, y=y, color=None if color=="None" else color,
                         title=f"{y} vs {x} Scatter", labels={x: x, y: y, "color": color})
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Histogram":
        col = st.selectbox("Column", nums)
        color = st.selectbox("Color By", ["None"] + cats)
        fig = px.histogram(df, x=col, color=None if color=="None" else color,
                           title=f"Histogram of {col}", labels={col: col})
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Box":
        y = st.selectbox("Y-axis", nums)
        x = st.selectbox("X-axis", ["None"] + cats)
        fig = px.box(df, y=y, x=None if x=="None" else x, points="all",
                     title=f"Box Plot of {y}" + (f" by {x}" if x!="None" else ""),
                     labels={y: y, x: x})
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Bar":
        x = st.selectbox("X-axis", cats)
        y = st.selectbox("Y-axis", nums)
        agg = st.selectbox("Aggregation", ["count", "mean", "sum"])
        data = df.groupby(x)[y].agg(agg).reset_index()
        fig = px.bar(data, x=x, y=y, title=f"{agg.title()} of {y} by {x}",
                     labels={x: x, y: f"{agg.title()} {y}"})
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Pie":
        cat = st.selectbox("Category", cats)
        data = df[cat].value_counts().reset_index()
        data.columns = [cat, "count"]
        fig = px.pie(data, names=cat, values="count", title=f"Pie Chart of {cat}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Violin":
        val = st.selectbox("Value", nums)
        cat = st.selectbox("Category", cats)
        fig = px.violin(df, y=val, x=cat, box=True, points="all", color=cat,
            title=f"Violin Plot of {val} by {cat}", labels={val: val, cat: cat})
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Line":
        x = st.selectbox("X-axis", nums + (["SurveyDate"] if "SurveyDate" in df.columns else []))
        y = st.selectbox("Y-axis", nums)
        fig = px.line(df, x=x, y=y, color_discrete_sequence=px.colors.sequential.Plasma, title=f"Line Plot of {y} over {x}", labels={x: x, y: y})
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Stacked Bar":
        cat1 = st.selectbox("Main Category", cats)
        cat2 = st.selectbox("Stacked by", [c for c in cats if c != cat1])
        data = df.groupby([cat1, cat2]).size().reset_index(name="count")
        fig = px.bar(data, x=cat1, y="count", color=cat2, barmode="stack", title=f"Stacked Bar of {cat1} by {cat2}",
                     labels={cat1: cat1, cat2: cat2, "count": "Count"})
        st.plotly_chart(fig, use_container_width=True)

# Classification Tab
elif page == "Classification":
    st.header("🧩 Classification")
    features = ["Age", "MonthlyDisposableIncome", "HealthConsciousness", "SpendPerServing"]
    label = safe_col(df, "Tried")
    if not label:
        st.error("Column 'Tried' not found in data.")
    else:
        df2 = df.copy()
        for col in features + [label]:
            if col in df2.columns and df2[col].dtype == "object":
                df2[col] = LabelEncoder().fit_transform(df2[col].astype(str))
        X = df2[[f for f in features if f in df2.columns]]
        y = df2[label]
        Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
        model = st.selectbox("Classifier", ["KNN", "Decision Tree", "Random Forest", "GBRT"])
        if model == "KNN":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif model == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=42)
        elif model == "Random Forest":
            clf = RandomForestClassifier(random_state=42)
        else:
            clf = GradientBoostingClassifier(random_state=42)
        clf.fit(Xt, yt)
        p = clf.predict(Xe)
        metrics = {
            "Accuracy": accuracy_score(ye, p),
            "Precision": precision_score(ye, p, zero_division=0),
            "Recall": recall_score(ye, p, zero_division=0),
            "F1": f1_score(ye, p, zero_division=0)
        }
        st.write("### Model Metrics")
        st.table(pd.DataFrame(metrics, index=["Value"]).T)
        cm = confusion_matrix(ye, p)
        z_text = [[str(y) for y in x] for x in cm]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
            colorscale="Blues", text=z_text, texttemplate="%{text}", showscale=True
        ))
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

# Clustering Tab (with Elbow Method)
elif page == "Clustering":
    st.header("🤝 Clustering")
    feats = ["Age", "MonthlyDisposableIncome", "SpendPerServing", "HealthConsciousness"]
    k = st.slider("Number of Clusters", 2, 8, 4)
    show_elbow = st.checkbox("Show Elbow Method (Optimal k)", value=True)
    if show_elbow:
        inertias = []
        ks = list(range(1, 11))
        for ki in ks:
            km = KMeans(n_clusters=ki, random_state=42).fit(df[feats])
            inertias.append(km.inertia_)
        fig_elbow = px.line(x=ks, y=inertias, markers=True,
            labels={"x": "Number of Clusters (k)", "y": "Inertia (SSE)"},
            title="Elbow Method for K Selection")
        st.plotly_chart(fig_elbow, use_container_width=True)
    km = KMeans(n_clusters=k, random_state=42).fit(df[feats])
    df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig = px.bar(centers, x=centers.index, y=feats, barmode="group",
                 color_discrete_sequence=px.colors.qualitative.Plotly,
                 title="Cluster Centers (Personas)",
                 labels={"index": "Cluster"})
    st.plotly_chart(fig, use_container_width=True)

# [Add more tabs as needed with same robust structure]
