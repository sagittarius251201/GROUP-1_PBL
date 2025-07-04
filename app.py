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

# --------- THEME LOGIC -------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")

theme = st.sidebar.radio("Theme", ['Light','Dark'], index=0)
st.session_state.theme = theme

if theme == 'Dark':
    bg_color = "#1B1B2F"
    fg_color = "#FFFFFF"
    sidebar_bg = "#30475E"
    highlight_color = "#EF476F"
    plotly_template = "plotly_dark"
else:
    bg_color = "#FFFFFF"
    fg_color = "#000000"
    sidebar_bg = "#F8F9FB"
    highlight_color = "#EF476F"
    plotly_template = "plotly_white"

st.markdown(f"""
    <style>
    body {{background-color:{bg_color}; color:{fg_color};}}
    .sidebar .sidebar-content {{background-color:{sidebar_bg};}}
    h1,h2,h3,h4 {{color:{highlight_color};}}
    </style>
""", unsafe_allow_html=True)

logo = st.sidebar.file_uploader("Upload Logo", type=["png","jpg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

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
        if isinstance(g, list) and len(g) > 0:
            filters.append("Gender: " + ", ".join([str(i) for i in g]))
    if "Occupation" in df:
        occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
        df = df[df.Occupation.isin(occ)]
        if isinstance(occ, list) and len(occ) > 0:
            filters.append("Occ: " + ", ".join([str(i) for i in occ]))

with st.sidebar.expander("Behavior"):
    if "ExerciseFrequency" in df:
        ex = st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
        df = df[df.ExerciseFrequency.isin(ex)]
        if isinstance(ex, list) and len(ex) > 0:
            filters.append("ExFreq: " + ", ".join([str(i) for i in ex]))
    if "ConsumptionFrequency" in df:
        cf = st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
        df = df[df.ConsumptionFrequency.isin(cf)]
        if isinstance(cf, list) and len(cf) > 0:
            filters.append("ConFreq: " + ", ".join([str(i) for i in cf]))

with st.sidebar.expander("Subscription & Location"):
    if "SubscribePlan" in df:
        sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
        df = df[df.SubscribePlan.isin(sub)]
        if isinstance(sub, list) and len(sub) > 0:
            filters.append("SubPlan: " + ", ".join([str(i) for i in sub]))
    if "City" in df:
        city = st.multiselect("City", df.City.unique(), df.City.unique())
        df = df[df.City.isin(city)]
        if isinstance(city, list) and len(city) > 0:
            filters.append("City: " + ", ".join([str(i) for i in city]))

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

if filters:
    chips = " ".join([f"<span style='background:{highlight_color};color:#FFF;padding:4px 8px;border-radius:4px;margin:2px'>{f}</span>" for f in filters])
    st.markdown("**Active Filters:** " + chips, unsafe_allow_html=True)

pages = [
    "Visualization","Classification","Clustering","Association","Anomaly","Regression",
    "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Changelog"
]
icons = [
    "bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
    "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","file-text"
]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal", styles={"container":{"padding":"0px"},"nav-link":{"font-size":"14px"},"nav-link-selected":{"background-color":highlight_color}})
st.markdown(f"**Home** > **{page}**")

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
                       opacity=0.7, color_discrete_sequence=px.colors.sequential.Plasma, template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Higher values of {y} are associated with higher {x} for {cats[0] if cats else 'all'} groups. Target users with high {x} for premium offerings.")
    elif chart=="Histogram":
        col=st.selectbox("Feature", nums)
        fig=px.histogram(df, x=col, nbins=30, color_discrete_sequence=px.colors.sequential.Viridis, template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
        most = df[col].mean()
        st.markdown(f"**Insight:** The average {col} is {most:.2f}. Use this value for targeting and pricing decisions.")
    elif chart=="Box":
        cat=st.selectbox("Category", cats)
        val=st.selectbox("Value", nums)
        fig=px.box(df, x=cat, y=val, color=cat, color_discrete_sequence=px.colors.sequential.Inferno, template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
        out = df[val][df[val]>df[val].quantile(0.75)].count()
        st.markdown(f"**Insight:** There are {out} above-average spenders in {cat}. Loyalty offers may be targeted here.")
    else:
        cat=st.selectbox("Category", cats)
        metric=st.selectbox("Metric", ["Count","Mean Spend","Sum Spend"])
        if metric=="Count":
            data=df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig=px.bar(data, x=cat, y="count", color_discrete_sequence=px.colors.sequential.Magma, template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            top = data.iloc[0][cat]
            st.markdown(f"**Insight:** The largest group is {top}. Tailor communication and distribution to this group.")
        else:
            agg=df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig=px.bar(agg, x=cat, y="SpendPerServing", color_discrete_sequence=px.colors.sequential.Magma, template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            hi = agg.sort_values("SpendPerServing", ascending=False).iloc[0][cat]
            st.markdown(f"**Insight:** {hi} segment drives the most spend. Prioritize in promotions.")

elif page=="Classification":
    st.header("🤖 Classification")
    X=df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y=LabelEncoder().fit_transform(df.TryNewBrand)
    Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler().fit(Xt); Xt_s,Xe_s=sc.transform(Xt),sc.transform(Xe)
    from sklearn.neighbors import KNeighborsClassifier
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
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted No","Predicted Yes"], 
            y=["Actual No","Actual Yes"],
            colorscale="Plasma",
            showscale=True,
            text=cm, texttemplate="%{text}", 
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template=plotly_template,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    fpr,tpr,_=roc_curve(ye,prob)
    fig2=go.Figure(go.Scatter(x=fpr,y=tpr,mode='lines', line_color=highlight_color))
    fig2.update_layout(template=plotly_template, title="ROC Curve")
    st.plotly_chart(fig2, use_container_width=True)
    best_metric = max(mets, key=mets.get)
    st.markdown(f"**Insight:** {algo} model achieves the highest {best_metric} ({mets[best_metric]:.2f}). Use this for predicting who will try the product.")

elif page=="Clustering":
    st.header("🤝 Clustering")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    st.subheader("K-Means Elbow Method")
    # Elbow method
    inertias = []
    k_range = range(2, 11)
    for ki in k_range:
        km_tmp = KMeans(n_clusters=ki, random_state=42).fit(df[feats])
        inertias.append(km_tmp.inertia_)
    fig_elbow = px.line(
        x=list(k_range), y=inertias, 
        markers=True, template=plotly_template,
        labels={'x':'Number of Clusters (K)','y':'Inertia'},
        title="Elbow Method for Optimal K"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.markdown("**Insight:** The 'elbow' point suggests the optimal number of clusters for customer segmentation.")

    # Main KMeans
    k=st.slider("Clusters (K)",2,8,4)
    km=KMeans(n_clusters=k,random_state=42).fit(df[feats])
    df["Cluster"]=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=feats).round(2)
    fig=px.bar(centers, x=centers.index, y=feats, barmode="group",
               color_discrete_sequence=px.colors.qualitative.Plotly, template=plotly_template)
    st.plotly_chart(fig,use_container_width=True)
    sizes = df['Cluster'].value_counts().sort_values(ascending=False)
    top_c = sizes.index[0]
    st.markdown(f"**Insight:** Cluster {top_c} is the largest segment with {sizes.iloc[0]} consumers. Use this persona for new product launches.")

# ...the rest of your code for other tabs stays the same, just be sure to add template=plotly_template in all plotly chart calls...
