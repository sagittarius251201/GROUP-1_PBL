
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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
    confusion_matrix, roc_curve,
    mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Super Enhanced v3 (Fixed)")

# Sidebar
st.sidebar.header("1. Data & Filters")
source = st.sidebar.radio("Load data via", ["GitHub URL", "Upload CSV"])
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
        st.sidebar.info("Upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(upload)

# Filters
st.sidebar.markdown("## Demographics")
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
df = df[df.Age.between(age_range[0], age_range[1])]

genders = st.sidebar.multiselect("Gender", df.Gender.unique(), default=list(df.Gender.unique()))
df = df[df.Gender.isin(genders)]

occupations = st.sidebar.multiselect("Occupation", df.Occupation.unique(), default=list(df.Occupation.unique()))
df = df[df.Occupation.isin(occupations)]

st.sidebar.markdown("## Behavior & Preferences")
cons_freq = st.sidebar.multiselect(
    "Consumption Frequency",
    df.ConsumptionFrequency.unique(),
    default=list(df.ConsumptionFrequency.unique())
)
df = df[df.ConsumptionFrequency.isin(cons_freq)]

channels = st.sidebar.multiselect(
    "Purchase Channel",
    df.PurchaseChannel.unique(),
    default=list(df.PurchaseChannel.unique())
)
df = df[df.PurchaseChannel.isin(channels)]

benefits = st.sidebar.multiselect(
    "Top Health Benefit",
    df.TopHealthBenefit.unique(),
    default=list(df.TopHealthBenefit.unique())
)
df = df[df.TopHealthBenefit.isin(benefits)]

packaging = st.sidebar.multiselect(
    "Packaging Format",
    df.PackagingFormat.unique(),
    default=list(df.PackagingFormat.unique())
)
df = df[df.PackagingFormat.isin(packaging)]

# Download filtered data
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("📥 Download Filtered Data", csv, "filtered_survey.csv")

# Key metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Spend (AED)", f"{df.SpendPerServing.mean():.2f}",
            help="Average spend per serving; higher suggests premium acceptance.")
col2.metric("Try New Brand %", f"{(df.TryNewBrand=='Yes').mean()*100:.1f}%",
            help="Rate willing to try a new brand; higher indicates interest.")
col3.metric("Subscription Interest %", f"{(df.SubscribePlan=='Yes').mean()*100:.1f}%",
            help="Interested in subscription; indicates recurring revenue potential.")
col4.metric("Avg Health Consciousness", f"{df.HealthConsciousness.mean():.1f}",
            help="Average health focus (1–5 scale).")

# Tabs
tabs = st.tabs(["Visualization","Classification","Clustering","Association","Anomaly","Regression"])

# Visualization Tab
with tabs[0]:
    st.header("Interactive Visualizations")
    fig = px.scatter(df, x="Age", y="SpendPerServing", color="Gender",
                     title="Spend per Serving by Age & Gender", size="MonthlyDisposableIncome")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(df, x="MonthlyDisposableIncome", nbins=30,
                       title="Monthly Disposable Income Distribution")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(df, x="ConsumptionFrequency", y="SpendPerServing",
                 title="Spend per Serving by Consumption Frequency")
    st.plotly_chart(fig, use_container_width=True)
    prefs = df.PackagingFormat.value_counts()
    fig = px.bar(x=prefs.index, y=prefs.values,
                 labels={'x':'Packaging','y':'Count'},
                 title="Preferred Packaging Formats")
    st.plotly_chart(fig, use_container_width=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Viridis"))
    fig.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Classification Tab
with tabs[1]:
    st.header("Classification: Willingness to Try New Brand")
    X = df.select_dtypes(include=np.number).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    algo = st.selectbox("Choose Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
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
    y_prob = model.predict_proba(X_test_s)[:,1]
    st.subheader("Performance Metrics")
    st.write({"Accuracy": accuracy_score(y_test,y_pred),
              "Precision": precision_score(y_test,y_pred),
              "Recall": recall_score(y_test,y_pred),
              "F1": f1_score(y_test,y_pred)})
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=algo))
    roc_fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', title='ROC Curve')
    st.plotly_chart(roc_fig, use_container_width=True)

# Clustering Tab
with tabs[2]:
    st.header("Clustering & Segmentation")
    feats = ['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    data = df[feats]
    k = st.slider("K-Means: # Clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster'] = km.labels_
    from sklearn.decomposition import PCA
    comp = PCA(2).fit_transform(StandardScaler().fit_transform(data))
    pca_df = pd.DataFrame(comp, columns=['PC1','PC2'])
    pca_df['Cluster'] = km.labels_.astype(str)
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title='PCA Cluster Projection')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Cluster Profiles")
    st.write(df.groupby('Cluster')[feats].mean().round(2))

# Association Tab
with tabs[3]:
    st.header("Association Rule Mining")
    cols = [c for c in df.columns if c.startswith("Flavour_") or c.startswith("Context_")]
    min_sup = st.slider("Min Support",0.01,0.2,0.03)
    min_conf = st.slider("Min Confidence",0.1,0.7,0.3)
    freq = apriori(df[cols], min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    st.subheader("Top 10 Rules by Lift")
    fig = px.bar(rules.sort_values('lift', ascending=False).head(10), 
                 x='lift', y=cols[0], orientation='h',
                 title='Top Rules by Lift', hover_data=['support','confidence'])
    st.plotly_chart(fig, use_container_width=True)

# Anomaly Tab
with tabs[4]:
    st.header("Anomaly Detection")
    feats = ['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    iso = IsolationForest(contamination=0.05, random_state=42).fit(data)
    df['Anomaly'] = iso.predict(data)
    fig = px.scatter(df, x='MonthlyDisposableIncome', y='SpendPerServing', 
                     color=df['Anomaly'].map({1:'Normal', -1:'Anomaly'}),
                     title='Anomaly Detection')
    st.plotly_chart(fig, use_container_width=True)

# Regression Tab
with tabs[5]:
    st.header("Regression: Predict Spend per Serving")
    Xr = df[['MonthlyDisposableIncome','HealthConsciousness','Age']]
    yr = df['SpendPerServing']
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    mse = mean_squared_error(yte, pred)
    rmse = np.sqrt(mse)
    st.subheader("Metrics")
    st.write({"R2": r2_score(yte,pred), "RMSE": rmse})
    fig = px.scatter(x=yte, y=pred, labels={'x':'Actual','y':'Predicted'},
                     title='Actual vs Predicted Spend')
    st.plotly_chart(fig, use_container_width=True)
    res = yte - pred
    fig = px.histogram(res, nbins=30, title='Residual Distribution')
    st.plotly_chart(fig, use_container_width=True)
