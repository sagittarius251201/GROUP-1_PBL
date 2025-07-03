
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
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
from streamlit_option_menu import option_menu

# Styling
st.markdown("""<style>
body { background-color: #1B1B2F; color: #FFFFFF; }
.sidebar .sidebar-content { background-color: #30475E; }
h1, h2, h3, h4 { color: #EF476F; }
</style>""", unsafe_allow_html=True)

# Data load
local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
if local.exists():
    df = pd.read_csv(local, parse_dates=["SurveyDate"])
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["SurveyDate"])
    else:
        st.stop()

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Fully Featured")

# Sidebar filters
with st.sidebar.expander("Demographics", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
    if "Gender" in df:
        g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
        df = df[df.Gender.isin(g)]
with st.sidebar.expander("Behavior"):
    if "ExerciseFrequency" in df:
        ex = st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
        df = df[df.ExerciseFrequency.isin(ex)]
    if "ConsumptionFrequency" in df:
        cf = st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
        df = df[df.ConsumptionFrequency.isin(cf)]
with st.sidebar.expander("Subscription & Location"):
    if "SubscribePlan" in df:
        sp = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
        df = df[df.SubscribePlan.isin(sp)]
    if "City" in df:
        city = st.multiselect("City", df.City.unique(), df.City.unique())
        df = df[df.City.isin(city)]
with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
st.sidebar.download_button("Download CSV", df.to_csv(index=False).encode(), "filtered.csv")

# Navigation
page = option_menu(None,
    ["Visualization","Classification","Clustering","Association","Anomaly","Regression",
     "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity"],
    icons=["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
           "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin"],
    orientation="horizontal", menu_icon="cast", default_index=0,
    styles={"container":{"background-color":"#1B1B2F"},
            "nav-link":{"color":"#FFFFFF","font-size":"16px","margin":"0px 8px"},
            "nav-link-selected":{"background-color":"#EF476F","color":"#FFFFFF"}}
)

# Visualization
if page=="Visualization":
    st.header("📊 Visualization")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if chart=="Scatter":
        x = st.selectbox("X-axis", nums, 0)
        y = st.selectbox("Y-axis", nums, 1)
        fig = px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                         color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
    elif chart=="Histogram":
        col = st.selectbox("Feature", nums)
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    elif chart=="Box":
        cat = st.selectbox("Category", cats)
        val = st.selectbox("Value", nums)
        fig = px.box(df, x=cat, y=val, color=cat, color_discrete_sequence=px.colors.sequential.Inferno)
        st.plotly_chart(fig, use_container_width=True)
    else:
        cat = st.selectbox("Category", cats)
        metric = st.selectbox("Metric", ["Count","Mean Spend","Sum Spend"])
        if metric=="Count":
            data = df[cat].value_counts().reset_index()
            data.columns=[cat,"count"]
            fig = px.bar(data, x=cat, y="count", color=cat, color_discrete_sequence=px.colors.sequential.Magma)
        else:
            agg = df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", color=cat, color_discrete_sequence=px.colors.sequential.Magma)
        st.plotly_chart(fig, use_container_width=True)

# Classification
elif page=="Classification":
    st.header("🤖 Classification")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s,Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (KNeighborsClassifier() if algo=="KNN" else
             DecisionTreeClassifier() if algo=="Decision Tree" else
             RandomForestClassifier() if algo=="Random Forest" else
             GradientBoostingClassifier())
    model.fit(Xtr_s,ytr); pred=model.predict(Xte_s); prob=model.predict_proba(Xte_s)[:,1]
    metrics = {
        "Accuracy": accuracy_score(yte,pred),
        "Precision": precision_score(yte,pred),
        "Recall": recall_score(yte,pred),
        "F1": f1_score(yte,pred)
    }
    st.table(pd.DataFrame(metrics, index=["Value"]).T)
    cm = confusion_matrix(yte,pred)
    fig = go.Figure(go.Heatmap(z=cm, x=["No","Yes"], y=["No","Yes"], colorscale="Plasma"))
    st.plotly_chart(fig, use_container_width=True)
    fpr,tpr,_ = roc_curve(yte,prob)
    fig2 = go.Figure(go.Scatter(x=fpr,y=tpr,mode="lines",line_color="#EF476F"))
    st.plotly_chart(fig2, use_container_width=True)

# Clustering
elif page=="Clustering":
    st.header("🤝 Clustering")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    k=st.slider("Clusters",2,8,4)
    km=KMeans(n_clusters=k,random_state=42).fit(df[feats])
    df["Cluster"]=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=feats).round(2)
    fig=px.bar(centers,x=centers.index,y=feats,barmode="group",
               color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig,use_container_width=True)

# Association
elif page=="Association":
    st.header("🔗 Association Rules")
    cols=[c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    sup=st.slider("Support",0.01,0.2,0.05)
    conf=st.slider("Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=sup,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=conf)
    rules["rule"]=rules["antecedents"].apply(lambda x:", ".join(x))+" → "+rules["consequents"].apply(lambda x:", ".join(x))
    top=rules.sort_values("lift",ascending=False).head(10)
    fig=px.bar(top,x="lift",y="rule",orientation="h",
               color="confidence",color_continuous_scale="Plasma")
    st.plotly_chart(fig,use_container_width=True)

# Anomaly
elif page=="Anomaly":
    st.header("🚨 Anomaly Detection")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso=IsolationForest(contamination=0.05,random_state=42).fit(df[feats])
    df["Anomaly"]=iso.predict(df[feats])
    fig=px.scatter(df,x="MonthlyDisposableIncome",y="SpendPerServing",
                   color=df["Anomaly"].map({1:"Normal",-1:"Anomaly"}),
                   color_discrete_sequence=["#00FF00","#FF0000"])
    st.plotly_chart(fig,use_container_width=True)

# Regression
elif page=="Regression":
    st.header("📈 Regression Comparison")
    Xr=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]; yr=df["SpendPerServing"]
    Xt,Xte,yt,yte=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    models = {
        "Linear": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "Tree": DecisionTreeRegressor(random_state=42)
    }
    results = []
    for name,mdl in models.items():
        mdl.fit(Xt,yt); pred=mdl.predict(Xte)
        results.append({
            "Model": name,
            "R2": r2_score(yte,pred),
            "RMSE": np.sqrt(mean_squared_error(yte,pred))
        })
    res_df=pd.DataFrame(results)
    st.table(res_df)
    fig=px.bar(res_df, x="Model", y=["R2","RMSE"], barmode="group",
               color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig,use_container_width=True)

# Forecasting
elif page=="Forecasting":
    st.header("⏱️ Forecasting")
    ts=df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    m=Prophet(); m.fit(ts)
    fut=m.make_future_dataframe(periods=30); fc=m.predict(fut)
    st.plotly_chart(plot_plotly(m,fc),use_container_width=True)

# Cohort
elif page=="Cohort":
    st.header("👥 Cohort Analysis")
    df["Cohort"]=df.SurveyDate.dt.to_period("M").astype(str)
    cohort=df.groupby("Cohort")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
    fig=px.line(cohort,x="Cohort",y="Rate",markers=True)
    st.plotly_chart(fig,use_container_width=True)

# Geography
elif page=="Geography":
    st.header("🗺️ Average Spend Per City")
    if "City" in df.columns and not df["City"].isnull().all():
        city_stats = df.groupby("City")["SpendPerServing"].mean().reset_index().sort_values("SpendPerServing", ascending=False)
        fig = px.bar(city_stats, x="City", y="SpendPerServing",
                     color="SpendPerServing",
                     color_continuous_scale="Viridis",
                     labels={"SpendPerServing": "Avg Spend (AED)"},
                     title="Average Spend by City")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Showing the average spend per serving in each city (UAE).")
    else:
        st.warning("No city data available in this dataset.")


# Sentiment
elif page=="Sentiment":
    st.header("💬 Sentiment Word Cloud")
    text=" ".join(df.Feedback.astype(str))
    wc=WordCloud(width=800,height=400).generate(text)
    plt.imshow(wc,interpolation="bilinear"); plt.axis("off")
    st.pyplot(plt.gcf())

# LTV & Churn
elif page=="LTV & Churn":
    st.header("💰 LTV & Churn")
    df["FreqNum"]=df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"]=df.SpendPerServing*df.FreqNum*12
    churn=(df.SubscribePlan=="No").astype(int)
    Xc=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt,Xv,yt,yv=train_test_split(Xc,churn,test_size=0.2,random_state=42)
    clf=RandomForestClassifier(random_state=42).fit(Xt,yt); pred=clf.predict(Xv)
    st.table(pd.DataFrame([{
        "Accuracy":accuracy_score(yv,pred),
        "Precision":precision_score(yv,pred),
        "Recall":recall_score(yv,pred)
    }]).T)
    st.plotly_chart(px.histogram(df,x="LTV",nbins=30),use_container_width=True)

# Price Elasticity
elif page=="Price Elasticity":
    st.header("💵 Price Elasticity")
    price=st.slider("Price per Serving",5,30,12)
    buyers=df[df.SpendPerServing>=price].shape[0]
    revenue=buyers*price
    st.metric("Estimated Buyers",buyers)
    st.metric("Projected Revenue (AED)",revenue)
