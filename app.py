import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
    mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Dashboard (Industry v4)")

# Sidebar: Data source & filters
st.sidebar.header("Data & Filters")
source = st.sidebar.selectbox("Load data via", ["GitHub URL", "Upload CSV"])
if source == "GitHub URL":
    default_url = "https://raw.githubusercontent.com/sagittarius251201/new-app/main/health_drink_survey_1000_responses.csv"
    url = st.sidebar.text_input("GitHub CSV URL", value=default_url)
    df = pd.read_csv(url)
else:
    up = st.sidebar.file_uploader("Upload CSV", type="csv")
    if up:
        df = pd.read_csv(up)
    else:
        st.sidebar.info("Upload CSV to proceed")
        st.stop()

# Filters
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
gf = st.sidebar.multiselect("Gender", df.Gender.unique(), default=df.Gender.unique())
df = df[(df.Age.between(*age_range)) & (df.Gender.isin(gf))]

tabs = st.tabs(["Visualization","Classification","Clustering","Assoc Rules","Anomaly","Regression"])

# Visualization
with tabs[0]:
    st.header("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(df.Age, df.SpendPerServing, alpha=0.6)
    ax.set_xlabel("Age"); ax.set_ylabel("Spend per Serving (AED)")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    df.MonthlyDisposableIncome.hist(bins=30, ax=ax)
    ax.set_title("Income Distribution"); st.pyplot(fig)

# Classification
with tabs[1]:
    st.header("Classification")
    X = df.select_dtypes(include=np.number).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    sel = st.selectbox("Algo", ["KNN","DT","RF","GBRT"])
    if sel=="KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif sel=="DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif sel=="RF":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:,1]

    metrics = {
        "Accuracy": accuracy_score(y_test,y_pred),
        "Precision": precision_score(y_test,y_pred),
        "Recall": recall_score(y_test,y_pred),
        "F1": f1_score(y_test,y_pred)
    }
    st.write(metrics)
    cm = confusion_matrix(y_test,y_pred)
    st.write("Confusion Matrix:", cm)

# Clustering
with tabs[2]:
    st.header("Clustering")
    feats = ['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    data = df[feats]
    k = st.slider("KMeans k",2,10,4)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster']=km.labels_
    st.write(df.groupby('Cluster')[feats].mean().round(2))

# Association Rules
with tabs[3]:
    st.header("Association Rules")
    cols = [c for c in df if c.startswith("Flavour_") or c.startswith("Context_")]
    sup = st.slider("Min Support",0.01,0.2,0.03)
    conf = st.slider("Min Confidence",0.1,0.7,0.3)
    freq = apriori(df[cols],min_support=sup,use_colnames=True)
    rules = association_rules(freq,metric="confidence",min_threshold=conf)
    st.write(rules.sort_values("lift",ascending=False).head(10))

# Anomaly
with tabs[4]:
    st.header("Anomaly Detection")
    iso = IsolationForest(contamination=0.05,random_state=42).fit(data)
    df['Anom']=iso.predict(data)
    st.write(df[df.Anom==-1].head())

# Regression
with tabs[5]:
    st.header("Regression")
    Xr = df[['MonthlyDisposableIncome','HealthConsciousness','Age']]
    yr = df['SpendPerServing']
    Xtr,Xte,ytr,yte = train_test_split(Xr,yr,test_size=0.2,random_state=42)
    reg = RandomForestRegressor(n_estimators=100,random_state=42).fit(Xtr,ytr)
    pred = reg.predict(Xte)
    st.write("R2:",r2_score(yte,pred),"RMSE:",mean_squared_error(yte,pred, squared=False))
