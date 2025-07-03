
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Enhanced UI/UX")

# Sidebar: Load and filter
st.sidebar.header("1. Load Data & Filters")
source = st.sidebar.radio("Data Source", ["GitHub URL", "Upload CSV"])
if source == "GitHub URL":
    default_url = "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv"
    url = st.sidebar.text_input("Raw GitHub CSV URL", default_url)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.sidebar.error(f"Could not load CSV:\n{e}")
        st.stop()
else:
    upload = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not upload:
        st.sidebar.info("Please upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(upload)

# Global filters
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
df = df[df.Age.between(age_range[0], age_range[1])]

genders = st.sidebar.multiselect("Gender", df.Gender.unique(), default=list(df.Gender.unique()))
df = df[df.Gender.isin(genders)]

occupations = st.sidebar.multiselect("Occupation", df.Occupation.unique(), default=list(df.Occupation.unique()))
df = df[df.Occupation.isin(occupations)]

cons_freq = st.sidebar.multiselect("Consumption Frequency", df.ConsumptionFrequency.unique(), default=list(df.ConsumptionFrequency.unique()))
df = df[df.ConsumptionFrequency.isin(cons_freq)]

channels = st.sidebar.multiselect("Purchase Channel", df.PurchaseChannel.unique(), default=list(df.PurchaseChannel.unique()))
df = df[df.PurchaseChannel.isin(channels)]

benefits = st.sidebar.multiselect("Top Health Benefit", df.TopHealthBenefit.unique(), default=list(df.TopHealthBenefit.unique()))
df = df[df.TopHealthBenefit.isin(benefits)]

packaging = st.sidebar.multiselect("Packaging Format", df.PackagingFormat.unique(), default=list(df.PackagingFormat.unique()))
df = df[df.PackagingFormat.isin(packaging)]

# Key metrics with tooltips
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    label="Avg Spend (AED)", 
    value=f"{df.SpendPerServing.mean():.2f}", 
    help="Mean spend per serving: indicates the average amount consumers are willing to pay. Higher suggests premium acceptance."
)
col2.metric(
    label="Try New Brand Rate", 
    value=f"{(df.TryNewBrand=='Yes').mean()*100:.1f}%", 
    help="Proportion of respondents willing to try a new UAE-made brand. Higher values signal strong market interest."
)
col3.metric(
    label="Subscription Interest", 
    value=f"{(df.SubscribePlan=='Yes').mean()*100:.1f}%", 
    help="Percentage interested in a monthly subscription plan: potential for recurring revenue models."
)
col4.metric(
    label="Avg Health Consciousness", 
    value=f"{df.HealthConsciousness.mean():.1f}", 
    help="Self-reported health focus on a 1–5 scale. Higher values mean a more health-aware audience."
)

# Tabs
tabs = st.tabs(["Visualization","Classification","Clustering","Association","Anomaly","Regression"])

# Visualization tab
with tabs[0]:
    st.header("Visualization")
    st.markdown("### Health Consciousness Distribution")
    fig, ax = plt.subplots()
    df.HealthConsciousness.value_counts(sort=False).plot.bar(ax=ax, color="#82E0AA")
    ax.set_xlabel("Level"); ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown("### Try New Brand by Occupation")
    rates = df.groupby('Occupation').TryNewBrand.apply(lambda x: (x=='Yes').mean())
    fig, ax = plt.subplots(figsize=(6,3))
    rates.sort_values().plot.barh(ax=ax, color="#85C1E9")
    ax.set_xlabel("Proportion Yes")
    st.pyplot(fig)

    st.markdown("### Consumption Frequency vs Spend")
    fig, ax = plt.subplots()
    df.boxplot(column='SpendPerServing', by='ConsumptionFrequency', ax=ax)
    ax.set_ylabel("Spend (AED)")
    st.pyplot(fig)

    st.markdown("### Packaging Preference Counts")
    fig, ax = plt.subplots()
    df.PackagingFormat.value_counts().plot.bar(ax=ax, color="#F7DC6F")
    ax.set_xlabel("Format"); ax.set_ylabel("Count")
    st.pyplot(fig)

# Classification tab
with tabs[1]:
    st.header("Classification")
    X = df.select_dtypes(include=np.number).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    if algo=="KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif algo=="Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif algo=="Random Forest":
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

# Clustering tab
with tabs[2]:
    st.header("Clustering")
    feats = ['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    data = df[feats]
    k = st.slider("K-Means: # Clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster'] = km.labels_
    st.subheader("Cluster Profiles")
    st.write(df.groupby('Cluster')[feats].mean().round(2))

# Association tab
with tabs[3]:
    st.header("Association Rules")
    cols = [c for c in df.columns if c.startswith("Flavour_") or c.startswith("Context_")]
    min_sup = st.slider("Min Support",0.01,0.2,0.03)
    min_conf = st.slider("Min Confidence",0.1,0.7,0.3)
    freq = apriori(df[cols], min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    st.subheader("Top 10 Rules by Lift")
    st.dataframe(rules.sort_values("lift",ascending=False).head(10))

# Anomaly tab
with tabs[4]:
    st.header("Anomaly Detection")
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = iso.fit_predict(df[['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']])
    st.write("Detected Anomalies")
    st.write(df[df.Anomaly==-1].head())

# Regression tab
with tabs[5]:
    st.header("Regression")
    Xr = df[['MonthlyDisposableIncome','HealthConsciousness','Age']]
    yr = df['SpendPerServing']
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    mse = mean_squared_error(yte,pred)
    rmse = np.sqrt(mse)
    st.subheader("Metrics")
    st.write({"R2": r2_score(yte, pred), "RMSE": rmse})
