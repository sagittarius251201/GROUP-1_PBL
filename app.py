import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
import time

# --- Session State & Config ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")

# --- Theme Switcher ---
theme = st.sidebar.radio("Theme", ['Light','Dark'], index=0)
st.session_state.theme = theme
if theme == 'Dark':
    st.markdown("""
        <style>
        body {background-color:#1B1B2F; color:#FFFFFF;}
        .sidebar .sidebar-content {background-color:#30475E;}
        h1,h2,h3,h4 {color:#EF476F;}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {background-color:#FFFFFF; color:#000000;}
        .sidebar .sidebar-content {background-color:#F8F9FB;}
        h1,h2,h3,h4 {color:#EF476F;}
        </style>
        """, unsafe_allow_html=True)

# --- Logo Upload ---
logo = st.sidebar.file_uploader("Upload Logo", type=["png","jpg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

# --- Data Loading & Refresh ---
def load_data():
    local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
    if local.exists():
        return pd.read_csv(local, parse_dates=["SurveyDate"])
    upload = st.sidebar.file_uploader("Upload data CSV", type="csv", key="data")
    if upload:
        return pd.read_csv(upload, parse_dates=["SurveyDate"])
    st.sidebar.error("Data file not found.")
    st.stop()

if st.sidebar.button("Refresh Data"):
    df = load_data()
    st.sidebar.write(f"Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

# --- Sidebar Filters & Chips ---
filters = []
with st.sidebar.expander("Demographics", expanded=True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    if "Gender" in df:
        g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
        df = df[df.Gender.isin(g)]
        filters.append("Gender: " + ", ".join(g))
    if "Occupation" in df:
        occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
        df = df[df.Occupation.isin(occ)]
        filters.append("Occ: " + ", ".join(occ))
with st.sidebar.expander("Behavior"):
    if "ExerciseFrequency" in df:
        ex = st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
        df = df[df.ExerciseFrequency.isin(ex)]
        filters.append("ExFreq: " + ", ".join(ex))
    if "ConsumptionFrequency" in df:
        cf = st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
        df = df[df.ConsumptionFrequency.isin(cf)]
        filters.append("ConFreq: " + ", ".join(cf))
with st.sidebar.expander("Subscription & Location"):
    if "SubscribePlan" in df:
        sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
        df = df[df.SubscribePlan.isin(sub)]
        filters.append("SubPlan: " + ", ".join(sub))
    if "City" in df:
        city = st.multiselect("City", df.City.unique(), df.City.unique())
        df = df[df.City.isin(city)]
        filters.append("City: " + ", ".join(city))
with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

if filters:
    chips = " ".join([f"<span style='background:#EF476F;color:#FFF;padding:4px 8px;border-radius:4px;margin:2px'>{f}</span>" for f in filters])
    st.markdown("**Active Filters:** " + chips, unsafe_allow_html=True)

# --- Breadcrumbs & Navigation ---
pages = [
    "Visualization","Classification","Clustering","Association","Anomaly","Regression",
    "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Changelog"
]
icons = [
    "bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
    "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","file-text"
]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal", styles={"container":{"padding":"0px"},"nav-link":{"font-size":"14px"},"nav-link-selected":{"background-color":"#EF476F"}})
st.markdown(f"**Home** > **{page}**")

# Visualization
if page=="Visualization":
    st.header("📊 Visualization")
    c1,c2,c3=st.columns(3)
    avg_spend = df.SpendPerServing.mean()
    c1.metric("Avg Spend", f"AED {avg_spend:.2f}")
    trial_rate = (df.TryNewBrand=='Yes').mean()*100
    c2.metric("Trial Rate", f"{trial_rate:.1f}%")
    nsubs = (df.SubscribePlan=='Yes').sum()
    c3.metric("Subscribers", f"{nsubs}")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if chart=="Scatter":
        x=st.selectbox("X-axis", nums, 0); y=st.selectbox("Y-axis", nums, 1)
        fig=px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                       opacity=0.7, color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Higher values of {y} are associated with higher {x} for {cats[0] if cats else 'all'} groups. Target users with high {x} for premium offerings.")
    elif chart=="Histogram":
        col=st.selectbox("Feature", nums)
        fig=px.histogram(df, x=col, nbins=30, color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
        most = df[col].mean()
        st.markdown(f"**Insight:** The average {col} is {most:.2f}. Use this value for targeting and pricing decisions.")
    elif chart=="Box":
        cat=st.selectbox("Category", cats)
        val=st.selectbox("Value", nums)
        fig=px.box(df, x=cat, y=val, color=cat, color_discrete_sequence=px.colors.sequential.Inferno)
        st.plotly_chart(fig, use_container_width=True)
        out = df[val][df[val]>df[val].quantile(0.75)].count()
        st.markdown(f"**Insight:** There are {out} above-average spenders in {cat}. Loyalty offers may be targeted here.")
    else:
        cat=st.selectbox("Category", cats)
        metric=st.selectbox("Metric", ["Count","Mean Spend","Sum Spend"])
        if metric=="Count":
            data=df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig=px.bar(data, x=cat, y="count", color_discrete_sequence=px.colors.sequential.Magma)
            st.plotly_chart(fig, use_container_width=True)
            top = data.iloc[0][cat]
            st.markdown(f"**Insight:** The largest group is {top}. Tailor communication and distribution to this group.")
        else:
            agg=df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig=px.bar(agg, x=cat, y="SpendPerServing", color_discrete_sequence=px.colors.sequential.Magma)
            st.plotly_chart(fig, use_container_width=True)
            hi = agg.sort_values("SpendPerServing", ascending=False).iloc[0][cat]
            st.markdown(f"**Insight:** {hi} segment drives the most spend. Prioritize in promotions.")

elif page=="Classification":
    st.header("🤖 Classification")
    X=df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y=LabelEncoder().fit_transform(df.TryNewBrand)
    Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler().fit(Xt); Xt_s,Xe_s=sc.transform(Xt),sc.transform(Xe)
    algo=st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (KNeighborsClassifier() if algo=="KNN" else
             DecisionTreeClassifier() if algo=="Decision Tree" else
             RandomForestClassifier() if algo=="Random Forest" else
             GradientBoostingClassifier())
    model.fit(Xt_s,yt); p=model.predict(Xe_s); prob=model.predict_proba(Xe_s)[:,1]
    mets={"Accuracy":accuracy_score(ye,p),"Precision":precision_score(ye,p),
          "Recall":recall_score(ye,p),"F1":f1_score(ye,p)}
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))
    cm=confusion_matrix(ye,p)
    fig=go.Figure(go.Heatmap(z=cm, x=["No","Yes"], y=["No","Yes"], colorscale="Plasma"))
    st.plotly_chart(fig, use_container_width=True)
    fpr,tpr,_=roc_curve(ye,prob)
    fig2=go.Figure(go.Scatter(x=fpr,y=tpr,mode='lines', line_color='#EF476F'))
    st.plotly_chart(fig2, use_container_width=True)
    best_metric = max(mets, key=mets.get)
    st.markdown(f"**Insight:** {algo} model achieves the highest {best_metric} ({mets[best_metric]:.2f}). Use this for predicting who will try the product.")

elif page=="Clustering":
    st.header("🤝 Clustering")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    k=st.slider("Clusters",2,8,4)
    km=KMeans(n_clusters=k,random_state=42).fit(df[feats])
    df["Cluster"]=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=feats).round(2)
    fig=px.bar(centers, x=centers.index, y=feats, barmode="group",
               color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig,use_container_width=True)
    sizes = df['Cluster'].value_counts().sort_values(ascending=False)
    top_c = sizes.index[0]
    st.markdown(f"**Insight:** Cluster {top_c} is the largest segment with {sizes.iloc[0]} consumers. Use this persona for new product launches.")

elif page=="Association":
    st.header("🔗 Association Rules")
    cols=[c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    s=st.slider("Min Support",0.01,0.2,0.05); c=st.slider("Min Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=s,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=c)
    rules["rule"]=rules["antecedents"].apply(lambda x:", ".join(x))+" → "+rules["consequents"].apply(lambda x:", ".join(x))
    top=rules.sort_values("lift",ascending=False).head(10)
    fig=px.bar(top,x="lift",y="rule",orientation="h", color="confidence", color_continuous_scale="Plasma")
    st.plotly_chart(fig,use_container_width=True)
    if not top.empty:
        top_rule = top.iloc[0]["rule"]
        st.markdown(f"**Insight:** Strongest association is: {top_rule}. Bundle these features/flavours in marketing.")

elif page=="Anomaly":
    st.header("🚨 Anomaly Detection")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso=IsolationForest(contamination=0.05,random_state=42).fit(df[feats])
    df["Anomaly"]=iso.predict(df[feats])
    fig=px.scatter(df,x="MonthlyDisposableIncome",y="SpendPerServing",
                   color=df["Anomaly"].map({1:"Normal",-1:"Anomaly"}),
                   color_discrete_sequence=["#00FF00","#FF0000"])
    st.plotly_chart(fig,use_container_width=True)
    outliers = (df["Anomaly"]==-1).sum()
    st.markdown(f"**Insight:** {outliers} anomalies detected. Review these cases for potential data errors or valuable edge customer behaviors.")

elif page=="Regression":
    st.header("📈 Regression Comparison")
    Xr=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]; yr=df["SpendPerServing"]
    Xt,Xe,yt,ye=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    models={"Linear":LinearRegression(),"Lasso":Lasso(),"Ridge":Ridge(),"Tree":DecisionTreeRegressor()}
    res=[]
    for name,mdl in models.items():
        mdl.fit(Xt,yt); pr=mdl.predict(Xe)
        res.append({"Model":name,"R2":r2_score(ye,pr),"RMSE":np.sqrt(mean_squared_error(ye,pr))})
    rdf=pd.DataFrame(res)
    st.table(rdf)
    fig=px.bar(rdf,x="Model",y=["R2","RMSE"],barmode="group",
               color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig,use_container_width=True)
    best = rdf.iloc[rdf['R2'].idxmax()]['Model']
    st.markdown(f"**Insight:** {best} regression gives the best fit for spend prediction. Use this to model expected sales for new cohorts.")

elif page=="Forecasting":
    st.header("⏱️ Forecasting")
    ts=df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    if len(ts)>30:
        m=Prophet(); m.fit(ts); fut=m.make_future_dataframe(periods=30); fc=m.predict(fut)
        st.plotly_chart(plot_plotly(m,fc),use_container_width=True)
        next_week = fc.iloc[-7:]['yhat'].mean()
        st.markdown(f"**Insight:** Expected avg spend next week: AED {next_week:.2f}. Use for inventory planning.")
    else:
        st.warning("Not enough data for time series forecast.")

elif page=="Cohort":
    st.header("👥 Cohort Analysis")
    df["Cohort"]=df.SurveyDate.dt.to_period("M").astype(str)
    cr=df.groupby("Cohort")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
    fig=px.line(cr,x="Cohort",y="Rate",markers=True); st.plotly_chart(fig,use_container_width=True)
    most = cr.sort_values("Rate", ascending=False).iloc[0]
    st.markdown(f"**Insight:** {most['Cohort']} cohort had highest subscription rate. Time campaigns with these patterns.")

elif page=="Geography":
    st.header("🗺️ Average Spend Per City")
    if "City" in df.columns and not df["City"].isnull().all():
        city_stats = df.groupby("City")["SpendPerServing"].mean().reset_index()
        city_stats = city_stats.sort_values("SpendPerServing", ascending=False)
        fig = px.bar(
            city_stats, x="City", y="SpendPerServing",
            color="SpendPerServing",
            color_continuous_scale="Viridis",
            labels={"SpendPerServing": "Avg Spend (AED)"},
            title="Average Spend by City"
        )
        st.plotly_chart(fig, use_container_width=True)
        # --- Dynamic Insight Section ---
        overall_avg = df["SpendPerServing"].mean()
        top_city = city_stats.iloc[0]
        low_city = city_stats.iloc[-1]
        pct_top = ((top_city["SpendPerServing"] - overall_avg) / overall_avg) * 100
        pct_low = ((low_city["SpendPerServing"] - overall_avg) / overall_avg) * 100
        insight = (
            f"**Insight:**\n"
            f"- **{top_city['City']}** has
