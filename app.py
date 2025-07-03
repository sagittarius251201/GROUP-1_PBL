
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Ultimate")

# Sidebar filters
st.sidebar.header("Filters & Data")
source = st.sidebar.radio("Load data via", ["GitHub URL", "Upload CSV"])
if source == "GitHub URL":
    url = st.sidebar.text_input("Raw GitHub CSV URL",
        "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv")
    try:
        df = pd.read_csv(url)
    except:
        st.sidebar.error("Failed to load data from URL")
        st.stop()
else:
    file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not file:
        st.sidebar.info("Upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(file)

# Common filters
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
df = df[df.Age.between(age_range[0], age_range[1])]
genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
df = df[df.Gender.isin(genders)]
occupations = st.sidebar.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
df = df[df.Occupation.isin(occupations)]

st.sidebar.download_button("Download filtered data", df.to_csv(index=False).encode(), "filtered.csv")

# Tabs
tabs = st.tabs(["Visualization","Classification","Clustering","Association","Anomaly","Regression"])

# Visualization tab with toggles
with tabs[0]:
    st.header("1️⃣ Visualization")
    st.markdown("Choose chart type and axes:")
    chart_type = st.selectbox("Chart type", ["Scatter", "Histogram", "Box", "Bar"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if chart_type == "Scatter":
        x = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index("Age"))
        y = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index("SpendPerServing"))
        fig = px.scatter(df, x=x, y=y, color="Gender",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title=f"{chart_type}: {y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Relationship between {x} and {y} shows ... (business implication).")
    elif chart_type == "Histogram":
        col = st.selectbox("Feature", numeric_cols, index=numeric_cols.index("MonthlyDisposableIncome"))
        fig = px.histogram(df, x=col, nbins=30,
                           color_discrete_sequence=["#2A9D8F"],
                           title=f"{chart_type}: Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** The distribution of {col} indicates ... (implication).")
    elif chart_type == "Box":
        cat = st.selectbox("Category", ["Gender","Occupation","ConsumptionFrequency"], index=0)
        val = st.selectbox("Value", numeric_cols, index=numeric_cols.index("SpendPerServing"))
        fig = px.box(df, x=cat, y=val,
                     color_discrete_sequence=["#E76F51"],
                     title=f"{chart_type}: {val} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Variance in {val} across {cat} shows ... (implication).")
    else:
        cat = st.selectbox("Category", ["PackagingFormat","PurchaseChannel","TopHealthBenefit"], index=0)
        counts = df[cat].value_counts()
        fig = px.bar(x=counts.index, y=counts.values,
                     color_discrete_sequence=["#264653"],
                     title=f"{chart_type}: {cat} counts")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Preferences for {cat} suggest ... (implication).")

# Classification
with tabs[1]:
    st.header("2️⃣ Classification")
    X = df.select_dtypes(include=np.number).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xt)
    Xt_s, Xv_s = scaler.transform(Xt), scaler.transform(Xv)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    if algo=="KNN": clf = KNeighborsClassifier()
    elif algo=="Decision Tree": clf = DecisionTreeClassifier()
    elif algo=="Random Forest": clf=RandomForestClassifier()
    else: clf=GradientBoostingClassifier()
    clf.fit(Xt_s, yt); yp = clf.predict(Xv_s); yp_prob=clf.predict_proba(Xv_s)[:,1]
    metrics = {"Accuracy":accuracy_score(yv,yp),"Precision":precision_score(yv,yp),
               "Recall":recall_score(yv,yp),"F1":f1_score(yv,yp)}
    st.subheader("Metrics")
    st.json(metrics, expanded=False)
    st.markdown("**Insight:** Precision & recall indicate model's balance between correctly identifying trials and avoiding false positives (e.g., campaigns).")

# Clustering
with tabs[2]:
    st.header("3️⃣ Clustering")
    feats=['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    data = df[feats]
    k = st.slider("Clusters (k)",2,8,4)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster'] = km.labels_
    st.subheader("Cluster Profiles")
    st.write(df.groupby('Cluster')[feats].mean().round(2))
    st.markdown("**Insight:** Segments reveal ... (e.g., high-income high-spend cluster for premium marketing).")

# Association
with tabs[3]:
    st.header("4️⃣ Association Rules")
    cols=[c for c in df if c.startswith("Flavour_") or c.startswith("Context_")]
    sup=st.slider("Min Support",0.01,0.2,0.05)
    conf=st.slider("Min Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=sup,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=conf)
    rules['rule']=rules['antecedents'].apply(lambda x:','.join(x))+"→"+rules['consequents'].apply(lambda x:','.join(x))
    top=rules.sort_values('lift',ascending=False).head(10)
    fig=px.bar(top, x='lift', y='rule', orientation='h',
               color_discrete_sequence=px.colors.qualitative.Set3,
               title="Top Rules by Lift")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** These rule associations guide product bundling and cross-promotions.")

# Anomaly
with tabs[4]:
    st.header("5️⃣ Anomaly Detection")
    feats=['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    iso=IsolationForest(contamination=0.05, random_state=42).fit(data)
    df['Anomaly'] = iso.predict(data)
    fig=px.scatter(df, x='MonthlyDisposableIncome', y='SpendPerServing',
                   color=df['Anomaly'].map({1:'Normal',-1:'Anomaly'}),
                   color_discrete_map={'Normal':'#2A9D8F','Anomaly':'#E76F51'},
                   title="Anomaly Detection")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Outliers may indicate high-value customers or data errors to investigate.")

# Regression
with tabs[5]:
    st.header("6️⃣ Regression")
    Xr=df[['MonthlyDisposableIncome','HealthConsciousness','Age']]; yr=df['SpendPerServing']
    Xt,Xv,yt,yv=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    reg=RandomForestRegressor(n_estimators=100,random_state=42).fit(Xt,yt); pr=reg.predict(Xv)
    mse=mean_squared_error(yv,pr); rmse=np.sqrt(mse)
    st.write({"R2":r2_score(yv,pr),"RMSE":rmse})
    fig=px.scatter(x=yv,y=pr,labels={'x':'Actual','y':'Predicted'},
                   title="Actual vs Predicted Spend",
                   color_discrete_sequence=['#264653'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Model explains variability well (R2), guiding pricing forecasts.")
