import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
    mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(page_title="Health Drink Dashboard", page_icon="🥤", layout="wide")

# Sidebar: Data source and filters common to all
st.sidebar.image("https://i.imgur.com/4R9YVZK.png", use_column_width=True)
st.sidebar.title("Data & Settings")
data_option = st.sidebar.radio("Data Source", ["GitHub URL", "Upload CSV"])
if data_option == "GitHub URL":
    default_url = "https://raw.githubusercontent.com/sagittarius251201/new-app/refs/heads/main/health_drink_survey_1000_responses%20(1).csv"
    url = st.sidebar.text_input("GitHub CSV URL", value=default_url)
    df = pd.read_csv(url)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.sidebar.warning("Upload a CSV to proceed")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("## Global Filters")
age_range = st.sidebar.slider("Age Range", int(df.Age.min()), int(df.Age.max()), (18, 80))
df = df[(df.Age >= age_range[0]) & (df.Age <= age_range[1])]

# Main area
st.title("🥤 Health Drink Consumer Survey Dashboard")
tabs = st.tabs(["📈 Visualization", "🧠 Classification", "🔮 Clustering",
                "🔗 Association Rules", "📊 Regression"])

# 1. Visualization
with tabs[0]:
    st.header("Data Visualization")
    st.markdown("Explore key distributions and relationships in the survey data.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Age'], bins=20, color="#73C6B6")
        ax.set_xlabel("Age")
        st.pyplot(fig)

    with col2:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['MonthlyDisposableIncome'], bins=20, color="#AED6F1")
        ax.set_xlabel("Monthly Income (AED)")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Spend per Serving")
        fig, ax = plt.subplots()
        ax.hist(df['SpendPerServing'], bins=20, color="#F1948A")
        ax.set_xlabel("Spend per Serving (AED)")
        st.pyplot(fig)

    with col4:
        st.subheader("Gender vs Spend")
        fig, ax = plt.subplots()
        df.boxplot(column='SpendPerServing', by='Gender', ax=ax, patch_artist=True,
                   boxprops=dict(facecolor="#A9DFBF"))
        ax.set_ylabel("Spend (AED)")
        st.pyplot(fig)

    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    cax = ax.matshow(corr, cmap="YlGnBu")
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)

# 2. Classification
with tabs[1]:
    st.header("Classification: Predict Willingness to Try New Brand")
    st.markdown("Select an algorithm and evaluate performance metrics.")

    X = df.select_dtypes(include=[np.number]).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df['TryNewBrand'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alg = st.selectbox("Algorithm", ["KNN", "Decision Tree", "Random Forest", "GBRT"])
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }
    model = models[alg]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    st.subheader("Metrics")
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    st.json(metrics)

    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="Oranges")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', color='white')
    st.pyplot(fig)

    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=alg, color="#117A65")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# 3. Clustering
with tabs[2]:
    st.header("Clustering: Customer Segments")
    st.markdown("Use K-Means to identify consumer personas.")

    features = ['Age', 'MonthlyDisposableIncome', 'SpendPerServing', 'HealthConsciousness']
    data = df[features]
    k = st.slider("Number of Clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster'] = km.labels_

    st.subheader("Elbow Method")
    sse = [KMeans(n_clusters=i, random_state=42).fit(data).inertia_ for i in range(1,11)]
    fig, ax = plt.subplots()
    ax.plot(range(1,11), sse, marker='o')
    st.pyplot(fig)

    st.subheader("Cluster Centers")
    centers = pd.DataFrame(km.cluster_centers_, columns=features).round(2)
    st.table(centers)

# 4. Association Rules
with tabs[3]:
    st.header("Association Rule Mining")
    st.markdown("Discover relationships between preferences.")

    cols = [c for c in df.columns if c.startswith('Flavour_') or c.startswith('Context_')]
    basket = df[cols]
    min_sup = st.slider("Min Support", 0.01, 0.1, 0.03)
    min_conf = st.slider("Min Confidence", 0.1, 0.5, 0.3)
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    st.dataframe(rules.sort_values('lift', ascending=False).head(10))

# 5. Regression
with tabs[4]:
    st.header("Regression: Spend Prediction")
    st.markdown("Model spend per serving based on demographics.")

    Xr = df[['MonthlyDisposableIncome', 'HealthConsciousness', 'Age']]
    yr = df['SpendPerServing']
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    selected = st.selectbox("Regressor", list(reg_models.keys()))
    reg = reg_models[selected]
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)

    st.subheader("Performance")
    perf = {
        "R2 Score": r2_score(yte, pred),
        "RMSE": mean_squared_error(yte, pred, squared=False)
    }
    st.json(perf)
