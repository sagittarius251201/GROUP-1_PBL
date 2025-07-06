# ── AUTO-INSTALL FALLBACK FOR OPENAI ─────────────────────────────────────────
import subprocess, sys
def ensure_package(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
ensure_package("openai")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from textblob import TextBlob
import networkx as nx
import openai
import time

# ── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown("""
    <style>
      body {font-family:'Helvetica Neue',Arial,sans-serif;}
      .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 16px;
        margin-bottom: 24px;
      }
      abbr { text-decoration: none; }
      abbr[title]:hover { cursor: help; border-bottom: 1px dotted #EF476F; }
    </style>
""", unsafe_allow_html=True)

# ── PAGE CONFIG & WELCOME ───────────────────────────────────────────────────
st.set_page_config(page_title="🚀 Health Drink Dashboard", layout="wide")
st.title("Welcome to the Health Drink Insights Dashboard! 🎉")
st.markdown(
    "Use the sidebar filters to hone in on your audience, then explore each tab "
    "for dynamic charts, deep analysis, and actionable insights."
)

# ── DATA UPLOAD & LOADING ────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    local = Path("data") / "health_drink_survey_1000_augmented.csv"
    if local.exists():
        return pd.read_csv(local, parse_dates=["SurveyDate"])
    url = (
        "https://raw.githubusercontent.com/"
        "sagittarius251201/second-/main/"
        "health_drink_survey_1000_augmented.csv"
    )
    try:
        return pd.read_csv(url, parse_dates=["SurveyDate"])
    except:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file, parse_dates=["SurveyDate"])
        st.error("No data found locally, on GitHub, or via upload.")
        st.stop()

df = load_data(uploaded)

# ── SIDEBAR FILTERS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")
    amin, amax = int(df.Age.min()), int(df.Age.max())
    age = st.slider("Age", amin, amax, (amin, amax))
    df = df[df.Age.between(*age)]

    gender = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(gender)]

    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]

    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]

    st.download_button("📥 Download Filtered Data", df.to_csv(index=False), "filtered.csv")

# ── TAB NAVIGATION ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "Visualization","Classification","Clustering","Association","Anomaly",
    "Regression","Forecasting","Cohort","Geography","Sentiment",
    "LTV & Churn","Price Elasticity","Chat","Glossary"
])

# ── 1) VISUALIZATION ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Key Metrics")
    c1, c2, c3 = st.columns(3)
    avg_spend = df.SpendPerServing.mean()
    trial_rate = (df.TryNewBrand=="Yes").mean()*100
    subscribers = (df.SubscribePlan=="Yes").sum()
    c1.metric("Avg Spend (AED)", f"{avg_spend:.2f}")
    c2.metric("Trial Rate (%)", f"{trial_rate:.1f}")
    c3.metric("Subscribers", subscribers)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Custom Chart & Insight")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if chart=="Scatter":
        x = st.selectbox("X-axis", nums, key="vx")
        y = st.selectbox("Y-axis", nums, key="vy")
        fig = px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                         opacity=0.7, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        corr = df[x].corr(df[y])
        st.info(f"**Insight:** Correlation between **{x}** & **{y}** is {corr:.2f}.")
    elif chart=="Histogram":
        col = st.selectbox("Feature", nums, key="vh")
        fig = px.histogram(df, x=col, nbins=30, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        skew = df[col].skew()
        st.info(f"**Insight:** **{col}** mean {df[col].mean():.2f}, skewness {skew:.2f}.")
    elif chart=="Box":
        cat = st.selectbox("Category", cats, key="vb_cat")
        val = st.selectbox("Value", nums, key="vb_val")
        fig = px.box(df, x=cat, y=val, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        q5, q95 = df[val].quantile([.05, .95])
        st.info(f"**Insight:** Outliers <{q5:.2f} or >{q95:.2f}.")
    else:
        cat = st.selectbox("Category", cats, key="vbar_cat")
        agg = df[cat].value_counts().reset_index()
        agg.columns=[cat,"count"]
        fig = px.bar(agg, x=cat, y="count", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        top = agg.iloc[0][cat]
        st.info(f"**Insight:** Largest: **{top}** ({agg.iloc[0]['count']} respondents).")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 2) CLASSIFICATION ────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Will-Try Classifier")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler().fit(Xt)
    Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "GBRT": GradientBoostingClassifier()
    }[algo]
    model.fit(Xt_s, yt)
    preds = model.predict(Xe_s)
    prob = model.predict_proba(Xe_s)[:,1]
    mets = {
        "Accuracy":accuracy_score(ye,preds),
        "Precision":precision_score(ye,preds),
        "Recall":recall_score(ye,preds),
        "F1 Score": f1_score(ye,preds)
    }
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))
    if hasattr(model,'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig = px.bar(imp, orientation='h', title="Feature Importances", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Insight:** Top feature **{imp.index[0]}** importance {imp.iloc[0]:.2f}.")
    cm = confusion_matrix(ye, preds)
    fig_cm = go.Figure(go.Heatmap(z=cm,
        x=["Pred No","Pred Yes"], y=["Actual No","Actual Yes"],
        text=cm, texttemplate="%{text}", colorscale="Blues"))
    fig_cm.update_layout(title="Confusion Matrix", template="plotly_white")
    st.plotly_chart(fig_cm, use_container_width=True)
    fpr, tpr, _ = roc_curve(ye, prob)
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', line_color="#EF476F"))
    fig_roc.update_layout(title="ROC Curve", template="plotly_white")
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 3) CLUSTERING ────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤝 K-Means Clustering")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    inertias = [KMeans(n_clusters=k, random_state=0).fit(df[feats]).inertia_ for k in range(2,11)]
    fig = px.line(x=list(range(2,11)), y=inertias, markers=True,
                  labels={'x':'k','y':'Inertia'}, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    k = st.slider("Choose k",2,8,4)
    km = KMeans(n_clusters=k, random_state=0).fit(df[feats])
    df["Cluster"] = km.labels_
    counts = df.Cluster.value_counts().sort_index()
    st.bar_chart(counts)
    st.info(f"**Insight:** Cluster sizes – {counts.to_dict()}")
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    st.data_editor(centers, num_rows="dynamic", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 4) ASSOCIATION ──────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔗 Association Rules")
    items = [c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    sup = st.slider("Min Support",0.01,0.2,0.05)
    conf = st.slider("Min Confidence",0.1,0.7,0.3)
    freq = apriori(df[items], min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    rules["rule"] = (
        rules.antecedents.apply(lambda x:", ".join(x))
        + " → "
        + rules.consequents.apply(lambda x:", ".join(x))
    )
    top10 = rules.sort_values("lift", ascending=False).head(10)
    st.write(top10[["rule","support","confidence","lift"]])
    best = top10.iloc[0]
    st.info(f"**Insight:** Strongest rule: **{best.rule}** (lift={best.lift:.2f}).")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 5) ANOMALY ───────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚨 Anomaly Detection")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso = IsolationForest(contamination=0.05, random_state=0).fit(df[feats])
    df["Anomaly"] = iso.predict(df[feats])
    fig = px.scatter(df, x="MonthlyDisposableIncome", y="SpendPerServing",
                     color=df.Anomaly.map({1:"Normal",-1:"Anomaly"}),
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    outliers = (df.Anomaly==-1).sum()
    st.info(f"**Insight:** Detected {outliers} anomalies ({outliers/len(df)*100:.1f}%).")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 6) REGRESSION ───────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Regression Comparison")
    Xr = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    yr = df.SpendPerServing
    Xt, Xe, yt, ye = train_test_split(Xr, yr, test_size=0.2, random_state=0)
    mods = {
        "Linear":LinearRegression(),
        "Lasso":Lasso(),
        "Ridge":Ridge(),
        "Tree":DecisionTreeRegressor()
    }
    results=[]
    for name, m in mods.items():
        m.fit(Xt,yt)
        pr=m.predict(Xe)
        results.append({
            "Model":name,
            "R2":r2_score(ye,pr),
            "RMSE":np.sqrt(mean_squared_error(ye,pr))
        })
    res_df=pd.DataFrame(results)
    st.table(res_df)
    best = res_df.loc[res_df.R2.idxmax()]
    st.info(f"**Insight:** Best model **{best.Model}** with R2={best.R2:.2f}.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 7) FORECASTING ──────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⏱️ Forecasting")
    ts = df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"})
    if len(ts)>30:
        m = Prophet(); m.fit(ts)
        fut = m.make_future_dataframe(periods=30)
        fc = m.predict(fut)
        st.plotly_chart(plot_plotly(m,fc), use_container_width=True)
        trend = fc.yhat.diff().mean()
        st.info(f"**Insight:** Avg daily trend {trend:.2f} AED.")
    else:
        st.warning("Not enough data (<30 points).")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 8) COHORT ───────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👥 Cohort Analysis")
    mode = st.radio("Type", ["Time","Cluster"], horizontal=True)
    if mode=="Time":
        df["Month"]=df.SurveyDate.dt.to_period("M").astype(str)
        cr = df.groupby("Month").SubscribePlan.apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.line(cr, x="Month", y="Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        peak = cr.loc[cr.Rate.idxmax()]
        st.info(f"**Insight:** Peak subscription in {peak.Month} ({peak.Rate:.1%}).")
    else:
        feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
        k = st.slider("Clusters",2,8,4)
        df["CohortCluster"] = KMeans(n_clusters=k,random_state=0).fit_predict(df[feats])
        cr = df.groupby("CohortCluster").SubscribePlan.apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.bar(cr, x="CohortCluster", y="Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        topc = cr.loc[cr.Rate.idxmax()]
        st.info(f"**Insight:** Cluster {topc.CohortCluster} likely to subscribe ({topc.Rate:.1%}).")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 9) GEOGRAPHY ────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺️ Geography")
    city_avg = df.groupby("City").SpendPerServing.mean().reset_index()
    fig = px.bar(city_avg, x="City", y="SpendPerServing", color="SpendPerServing", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    topc = city_avg.loc[city_avg.SpendPerServing.idxmax()]
    st.info(f"**Insight:** Highest avg spend in {topc.City} ({topc.SpendPerServing:.2f} AED).")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 10) SENTIMENT ───────────────────────────────────────────────────────────
with tabs[9]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Sentiment")
    text = " ".join(df.Feedback.fillna("").tolist())
    wc = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig)
    scores = df.Feedback.fillna("").apply(lambda x: TextBlob(x).sentiment.polarity)
    dist = scores.value_counts(bins=[-1,-0.1,0.1,1]).sort_index()
    fig2 = px.pie(names=["Negative","Neutral","Positive"], values=dist.values, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    major = ["Negative","Neutral","Positive"][np.argmax(dist.values)]
    st.info(f"**Insight:** Majority sentiment is **{major}**.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 11) LTV & CHURN ─────────────────────────────────────────────────────────
with tabs[10]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 LTV & Churn")
    df["FreqNum"] = df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"] = df.SpendPerServing * df.FreqNum * 12
    churn = (df.SubscribePlan=="No").astype(int)
    Xc = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt, Xe, yt, ye = train_test_split(Xc, churn, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(random_state=0).fit(Xt, yt)
    pr = clf.predict(Xe)
    mets = {"Accuracy":accuracy_score(ye,pr),"Precision":precision_score(ye,pr),"Recall":recall_score(ye,pr)}
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))
    avg_ltv = df.LTV.mean()
    st.info(f"**Insight:** Avg LTV = {avg_ltv:.2f} AED.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 12) PRICE ELASTICITY ─────────────────────────────────────────────────────
with tabs[11]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💵 Price Elasticity")
    min_price = int(df.SpendPerServing.min())
    max_price = int(df.SpendPerServing.max())
    median_price = int(df.SpendPerServing.median())
    price = st.slider("Price per Serving (AED)", min_value=min_price, max_value=max_price, value=median_price)
    buyers = df[df.SpendPerServing >= price].shape[0]
    revenue = buyers * price
    conversion = buyers / len(df)
    st.metric("Expected Buyers", buyers)
    st.metric("Expected Revenue (AED)", revenue)
    st.info(f"**Insight:** Conversion at {price} AED is {conversion:.1%}.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 13) CHAT ─────────────────────────────────────────────────────────────────
with tabs[12]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Ask GPT")
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{"role":"system","content":"You are a data analyst assistant."}]
    query = st.chat_input("Your question...")
    if query:
        st.session_state.msgs.append({"role":"user","content":query})
        openai.api_key = st.secrets.get("OPENAI_API_KEY","")
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.msgs
            )
            msg = resp["choices"][0]["message"]
            st.session_state.msgs.append({"role":msg["role"],"content":msg["content"]})
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
    for m in st.session_state.msgs:
        st.chat_message(m["role"]).write(m["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# ── 14) GLOSSARY ────────────────────────────────────────────────────────────
with tabs[13]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📚 Glossary")
    glossary = {
        "SpendPerServing":"AED spent per serving.",
        "HealthConsciousness":"Self-rated health importance (1–10).",
        "TryNewBrand":"Willingness to try new brand (Yes/No).",
        "SubscribePlan":"Subscription status.",
        "MonthlyDisposableIncome":"Disposable income per month (AED).",
        "Age":"Age of respondent.",
        "ExerciseFrequency":"Exercise frequency.",
        "ConsumptionFrequency":"Consumption frequency.",
        "Feedback":"Free-text user feedback."
    }
    for term, desc in glossary.items():
        st.markdown(f"**{term}** <abbr title='{desc}'>ℹ️</abbr>")
    st.markdown('</div>', unsafe_allow_html=True)
