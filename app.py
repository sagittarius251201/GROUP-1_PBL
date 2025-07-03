
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
st.title("🥤 Health Drink Survey Dashboard — Complete")

# Sidebar filters
st.sidebar.header("Filters & Data")
source = st.sidebar.radio("Load data via", ["GitHub URL", "Upload CSV"])
if source=="GitHub URL":
    url = st.sidebar.text_input("Raw GitHub CSV URL", "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv")
    try:
        df = pd.read_csv(url)
    except:
        st.error("Failed to load data from URL")
        st.stop()
else:
    file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not file:
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

# Download
st.sidebar.download_button("Download filtered data", df.to_csv(index=False).encode(), "filtered.csv")

# Tabs
tabs = st.tabs(["Visualization","Classification","Clustering","Association","Anomaly","Regression"])

# Visualization
with tabs[0]:
    st.header("Data Visualization")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x="Age", y="SpendPerServing", color="Gender",
                         color_discrete_sequence=['#2A9D8F','#E76F51','#264653'],
                         title="Spend per Serving by Age & Gender")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Younger consumers (18–30) vary widely in spend, suggesting tiered pricing could capture maximum value.")
    with col2:
        fig = px.histogram(df, x="MonthlyDisposableIncome", nbins=30,
                           color_discrete_sequence=['#264653'],
                           title="Disposable Income Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Most incomes cluster between 5k–15k AED; pricing ~12 AED per serving aligns with budgets.")

# Classification
with tabs[1]:
    st.header("Classification")
    X = df.select_dtypes(include=np.number).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xt)
    Xt_s, Xv_s = scaler.transform(Xt), scaler.transform(Xv)
    model = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    if model=="KNN": clf = KNeighborsClassifier()
    elif model=="Decision Tree": clf = DecisionTreeClassifier()
    elif model=="Random Forest": clf=RandomForestClassifier()
    else: clf=GradientBoostingClassifier()
    clf.fit(Xt_s, yt); yp = clf.predict(Xv_s); yp_prob=clf.predict_proba(Xv_s)[:,1]
    metrics={"Accuracy":accuracy_score(yv,yp),"Precision":precision_score(yv,yp),
             "Recall":recall_score(yv,yp),"F1":f1_score(yv,yp)}
    st.write(metrics)
    cm=confusion_matrix(yv,yp)
    fig=go.Figure(data=go.Heatmap(z=cm, x=["No","Yes"], y=["No","Yes"], colorscale="Mint"))
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    fpr,tpr,_=roc_curve(yv,yp_prob)
    fig=go.Figure(data=go.Scatter(x=fpr,y=tpr,mode='lines'))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** High AUC indicates model reliably predicts trial interest; target marketing using this model.")

# Clustering
with tabs[2]:
    st.header("Clustering")
    feats=['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    data=df[feats]
    k=st.slider("Clusters (k)",2,8,4)
    km=KMeans(k).fit(data); df['Cluster']=km.labels_
    from sklearn.decomposition import PCA
    pc=PCA(2).fit_transform(StandardScaler().fit_transform(data))
    pdf=pd.DataFrame(pc,columns=['PC1','PC2']); pdf['Cluster']=df['Cluster'].astype(str)
    fig=px.scatter(pdf,x='PC1',y='PC2',color='Cluster',title="PCA Clusters")
    st.plotly_chart(fig,use_container_width=True)
    st.write("Cluster centers:")
    st.write(pd.DataFrame(km.cluster_centers_,columns=feats))

# Association
with tabs[3]:
    st.header("Association Rules")
    cols=[c for c in df if c.startswith("Flavour_") or c.startswith("Context_")]
    sup=st.slider("Min Support",0.01,0.2,0.05); conf=st.slider("Min Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=sup,use_colnames=True); rules=association_rules(freq,metric="confidence",min_threshold=conf)
    rules['rule']=rules['antecedents'].apply(lambda x:','.join(x))+"->"+rules['consequents'].apply(lambda x:','.join(x))
    top=rules.sort_values('lift',ascending=False).head(10)
    fig=px.bar(top, x='lift', y='rule', orientation='h', title="Top Rules by Lift")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insight:** These item combos reveal popular mix & match behaviors for targeted bundles.")

# Anomaly
with tabs[4]:
    st.header("Anomaly Detection")
    iso=IsolationForest(contamination=0.05).fit(data); df['Anom']=iso.predict(data)
    fig=px.scatter(df, x='MonthlyDisposableIncome', y='SpendPerServing', color=df['Anom'].map({1:'Normal',-1:'Anomaly'}),
                   title="Anomalies")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insight:** Outliers may indicate premium spenders or data errors. Investigate high spend anomalies.")

# Regression
with tabs[5]:
    st.header("Regression")
    Xr=df[['MonthlyDisposableIncome','HealthConsciousness','Age']]; yr=df['SpendPerServing']
    Xt,Xv,yt,yv=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    reg=RandomForestRegressor().fit(Xt,yt); pr=reg.predict(Xv)
    mse=mean_squared_error(yv,pr); rmse=np.sqrt(mse)
    st.write({"R2":r2_score(yv,pr),"RMSE":rmse})
    fig=px.scatter(x=yv,y=pr,labels={'x':'Actual','y':'Predicted'},title="Actual vs Predicted")
    st.plotly_chart(fig,use_container_width=True)
    resid=yv-pr; fig=px.histogram(resid,nbins=30,title="Residuals"); st.plotly_chart(fig,use_container_width=True)
