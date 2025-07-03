import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
    mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Industry-Grade Health Drink Dashboard", page_icon="🥤", layout="wide")
st.title("🥤 Health Drink Survey Dashboard (Industry Grade)")

# Sidebar: Data source & filters
st.sidebar.header("1. Data Source & Filters")
data_source = st.sidebar.selectbox("Load data via", ["GitHub URL", "Upload CSV"])
if data_source == "GitHub URL":
    default_url = "https://raw.githubusercontent.com/sagittarius251201/new-app/refs/heads/main/health_drink_survey_1000_responses%20(1).csv"
    url = st.sidebar.text_input("GitHub raw CSV URL", value=default_url)
    df = pd.read_csv(url)
else:
    upload = st.sidebar.file_uploader("Upload CSV", type="csv")
    if upload:
        df = pd.read_csv(upload)
    else:
        st.sidebar.info("Awaiting CSV upload")
        st.stop()

# Global filters
st.sidebar.markdown("### Demographic Filters")
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
gender_sel = st.sidebar.multiselect("Gender", df.Gender.unique(), default=list(df.Gender.unique()))
flavours_sel = st.sidebar.multiselect("Flavour Prefs", [c for c in df.columns if c.startswith("Flavour_")], default=[c for c in df.columns if c.startswith("Flavour_")])
channel_sel = st.sidebar.multiselect("Purchase Channels", df.PurchaseChannel.unique(), default=list(df.PurchaseChannel.unique()))

filtered = df[
    (df.Age.between(age_range[0], age_range[1])) &
    (df.Gender.isin(gender_sel)) &
    (df.PurchaseChannel.isin(channel_sel))
]
for col in flavours_sel:
    filtered = filtered[filtered[col] == 1]

# Tabs
tabs = st.tabs(["📈 Visualization", "🧠 Classification", "🔮 Clustering", "🔗 Association Rules", "⚡ Anomaly Detection", "📊 Regression"])

# 1. Visualization
with tabs[0]:
    st.header("Interactive Data Visualization")
    st.markdown("Use the selectors above to filter and explore the data.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age vs Spend (Scatter)")
        fig = px.scatter(filtered, x="Age", y="SpendPerServing", color="Gender", size="MonthlyDisposableIncome",
                         title="Spend per Serving by Age & Income")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Income Distribution")
        fig = px.histogram(filtered, x="MonthlyDisposableIncome", nbins=30, title="Income Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Spend Distribution")
        fig = px.histogram(filtered, x="SpendPerServing", nbins=30, title="Spend per Serving")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("Taste Preferences")
        pref_cols = [c for c in filtered.columns if c.startswith("Flavour_")]
        prefs = filtered[pref_cols].sum().sort_values(ascending=False)
        fig = px.bar(x=prefs.index.str.replace("Flavour_", ""), y=prefs.values, title="Flavour Preference Counts")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Parallel Coordinates for Key Features")
    pc_df = filtered[["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]]
    fig = px.parallel_coordinates(pc_df, color="HealthConsciousness", labels={"MonthlyDisposableIncome": "Income", "SpendPerServing": "Spend"})
    st.plotly_chart(fig, use_container_width=True)

# 2. Classification
with tabs[1]:
    st.header("Classification: Willingness to Try New Brand")
    # Feature selection and encoding
    X = filtered.select_dtypes(include=np.number).drop(columns=["SpendPerServing"])
    y = LabelEncoder().fit_transform(filtered["TryNewBrand"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Select Classifier", ["KNN","Decision Tree","Random Forest","GBRT"])
    params = {}
    if model_choice == "KNN":
        params["n_neighbors"] = st.slider("KNN: Number of Neighbors", 1, 20, 5)
        clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    elif model_choice == "Decision Tree":
        params["max_depth"] = st.slider("DT: Max Depth", 1, 20, 5)
        clf = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=42)
    elif model_choice == "Random Forest":
        params["n_estimators"] = st.slider("RF: Number of Trees", 10, 200, 100, step=10)
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=42)
    else:
        params["learning_rate"] = st.slider("GBRT: Learning Rate", 0.01, 0.5, 0.1)
        params["n_estimators"] = st.slider("GBRT: Number of Estimators", 50, 200, 100, step=10)
        clf = GradientBoostingClassifier(learning_rate=params["learning_rate"], n_estimators=params["n_estimators"], random_state=42)

    clf.fit(X_train, y_train); y_pred = clf.predict(X_test); y_prob = clf.predict_proba(X_test)[:,1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    st.subheader("Performance Metrics")
    st.json(metrics)

    st.subheader("Feature Importance")
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig = px.bar(x=imp.index, y=imp.values, title="Feature Importances")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for selected model.")

with tabs[2]:
    st.header("Clustering & Segmentation")
    algo = st.selectbox("Clustering Algorithm", ["KMeans","DBSCAN","Agglomerative"])
    features = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    data = filtered[features]
    if algo == "KMeans":
        k = st.slider("KMeans: Number of Clusters", 2, 10, 4)
        clusterer = KMeans(n_clusters=k, random_state=42)
    elif algo == "DBSCAN":
        eps = st.slider("DBSCAN: Epsilon", 0.1, 5.0, 0.5)
        min_samples = st.slider("DBSCAN: Min Samples", 1, 20, 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        k = st.slider("Agglomerative: Number of Clusters", 2, 10, 4)
        clusterer = AgglomerativeClustering(n_clusters=k)
    labels = clusterer.fit_predict(data)
    filtered["Cluster"] = labels

    st.subheader("Segmentation Scatter (PCA)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    comps = pca.fit_transform(StandardScaler().fit_transform(data))
    pca_df = pd.DataFrame(comps, columns=["PC1","PC2"])
    pca_df["Cluster"] = labels.astype(str)
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="PCA Projection of Clusters")
    st.plotly_chart(fig, use_container_width=True)

# 4. Association Rules
with tabs[3]:
    st.header("Advanced Association Rule Mining")
    choices = {"Flavours": [c for c in filtered.columns if c.startswith("Flavour_")],
               "Contexts": [c for c in filtered.columns if c.startswith("Context_")],
               "Barriers": [c for c in filtered.columns if c.startswith("Barrier_")]}
    sel_group = st.radio("Select Item Group", list(choices.keys()))
    cols = choices[sel_group]
    min_sup = st.slider("Min Support", 0.01, 0.2, 0.05)
    min_conf = st.slider("Min Confidence", 0.1, 0.7, 0.3)
    freq = apriori(filtered[cols], min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    st.subheader("Top-10 Rules by Lift")
    st.dataframe(rules.sort_values("lift", ascending=False).head(10))

# 5. Anomaly Detection
with tabs[4]:
    st.header("Anomaly Detection: Isolation Forest")
    from sklearn.ensemble import IsolationForest
    cont = st.slider("Contamination Fraction", 0.01, 0.2, 0.05)
    iso = IsolationForest(contamination=cont, random_state=42)
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    pred = iso.fit_predict(filtered[feats])
    filtered["Anomaly"] = pred
    st.subheader("Anomalies in Spend vs Income")
    fig = px.scatter(filtered, x="MonthlyDisposableIncome", y="SpendPerServing",
                     color=filtered["Anomaly"].map({1:"Normal",-1:"Anomaly"}), title="Anomaly Detection")
    st.plotly_chart(fig, use_container_width=True)

# 6. Regression
with tabs[5]:
    st.header("Regression: Spend Prediction")
    Xr = filtered[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    yr = filtered["SpendPerServing"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GBRT": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    sel = st.selectbox("Select Regressor", list(reg_models.keys()))
    reg = reg_models[sel]
    reg.fit(Xtr, ytr)
    yp = reg.predict(Xte)
    st.subheader("Metrics")
    st.json({"R2": r2_score(yte, yp), "RMSE": mean_squared_error(yte, yp, squared=False)})
    st.subheader("Predicted vs Actual")
    fig = px.scatter(x=yte, y=yp, labels={"x":"Actual","y":"Predicted"}, title="Actual vs Predicted Spend")
    st.plotly_chart(fig, use_container_width=True)
