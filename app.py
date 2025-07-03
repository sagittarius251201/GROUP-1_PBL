import streamlit as st
import pandas as pd
import numpy as np
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
    confusion_matrix, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules

# --- Page config ---
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard")

# --- Sidebar: Data source & filters ---
st.sidebar.header("1. Load Data & Filters")

source = st.sidebar.radio("Data Source", ["GitHub URL", "Upload CSV"])
if source == "GitHub URL":
    default_url = "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv"
    url = st.sidebar.text_input("Raw GitHub CSV URL", default_url)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.sidebar.error(f"⚠️ Could not load CSV:\n{e}")
        st.stop()
else:
    upload = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not upload:
        st.sidebar.info("Please upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(upload)

# Demographic filters
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
df = df[df.Age.between(age_range[0], age_range[1])]

genders = st.sidebar.multiselect("Gender", df.Gender.unique(), default=list(df.Gender.unique()))
df = df[df.Gender.isin(genders)]

# --- Tabs ---
tabs = st.tabs([
    "📊 Visualization",
    "🤖 Classification",
    "🔍 Clustering",
    "🔗 Association Rules",
    "🚨 Anomaly Detection",
    "📈 Regression"
])

# 1. Visualization
with tabs[0]:
    st.header("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(df.Age, df.SpendPerServing, alpha=0.6)
    ax.set_xlabel("Age")
    ax.set_ylabel("Spend per Serving (AED)")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    df.MonthlyDisposableIncome.hist(bins=30, ax=ax)
    ax.set_title("Income Distribution")
    st.pyplot(fig)

# 2. Classification
with tabs[1]:
    st.header("Classification: Willingness to Try New Brand")
    X = df.select_dtypes(include=np.number).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    algo = st.selectbox("Choose Algorithm", ["KNN", "Decision Tree", "Random Forest", "GBRT"])
    if algo == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif algo == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif algo == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    st.subheader("Performance Metrics")
    st.write({
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

# 3. Clustering
with tabs[2]:
    st.header("Clustering & Segmentation")
    feats = ['Age', 'MonthlyDisposableIncome', 'SpendPerServing', 'HealthConsciousness']
    data = df[feats]
    k = st.slider("K-Means: # Clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster'] = km.labels_
    st.subheader("Cluster Centers (Mean Values)")
    st.write(df.groupby('Cluster')[feats].mean().round(2))

# 4. Association Rules
with tabs[3]:
    st.header("Association Rule Mining")
    cols = [c for c in df.columns if c.startswith("Flavour_") or c.startswith("Context_")]
    min_sup = st.slider("Min Support", 0.01, 0.2, 0.03)
    min_conf = st.slider("Min Confidence", 0.1, 0.7, 0.3)
    frequent = apriori(df[cols], min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    st.subheader("Top 10 Rules by Lift")
    st.dataframe(rules.sort_values("lift", ascending=False).head(10))

# 5. Anomaly Detection
with tabs[4]:
    st.header("Anomaly Detection (Isolation Forest)")
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = iso.fit_predict(df[['Age', 'MonthlyDisposableIncome', 'SpendPerServing', 'HealthConsciousness']])
    st.write("Sample Anomalies:")
    st.write(df[df.Anomaly == -1].head())

# 6. Regression
with tabs[5]:
    st.header("Regression: Predict Spend per Serving")
    Xr = df[['MonthlyDisposableIncome', 'HealthConsciousness', 'Age']]
    yr = df['SpendPerServing']
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)

    mse = mean_squared_error(yte, pred)
    rmse = np.sqrt(mse)
    st.subheader("Regression Metrics")
    st.write({
        "R2 Score": r2_score(yte, pred),
        "RMSE": rmse
    })
