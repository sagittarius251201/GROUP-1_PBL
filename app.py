import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
import time

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
      .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 16px;
        margin-bottom: 20px;
      }
      abbr { text-decoration: none; }
      abbr[title]:hover { cursor: help; border-bottom: 1px dotted #EF476F; }
    </style>
""", unsafe_allow_html=True)

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Health Drink Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    path = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["SurveyDate"])
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        return pd.read_csv(uploaded, parse_dates=["SurveyDate"])
    st.error("No data file found.")
    st.stop()

df = load_data()

# ── SIDEBAR FILTERS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")
    age_min, age_max = int(df.Age.min()), int(df.Age.max())
    age = st.slider("Age", age_min, age_max, (age_min, age_max))
    df = df[df.Age.between(*age)]

    gender = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(gender)]

    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]

    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]

    st.download_button("📥 Download CSV", df.to_csv(index=False), "filtered.csv", "text/csv")

# ── TAB NAVIGATION ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "Visualization","Classification","Clustering","Association",
    "Anomaly","Regression","Forecasting","Cohort",
    "Geography","Sentiment","LTV & Churn","Price Elasticity","Glossary"
])

# ── 1. VISUALIZATION ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Quick Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Spend", f"AED {df.SpendPerServing.mean():.2f}")
    c2.metric("Trial Rate", f"{(df.TryNewBrand=='Yes').mean()*100:.1f}%")
    c3.metric("Subscribers", f"{(df.SubscribePlan=='Yes').sum()}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Custom Chart")
    chart = st.selectbox("Choose visualization", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()

    if chart == "Scatter":
        x = st.selectbox("X-axis", nums, key="viz_x")
        y = st.selectbox("Y-axis", nums, key="viz_y")
        fig = px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                         opacity=0.7, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Histogram":
        col = st.selectbox("Feature", nums, key="viz_hist")
        fig = px.histogram(df, x=col, nbins=30, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Box":
        cat = st.selectbox("Category", cats, key="viz_box_cat")
        val = st.selectbox("Value", nums, key="viz_box_val")
        fig = px.box(df, x=cat, y=val, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        cat = st.selectbox("Category", cats, key="viz_bar_cat")
        data = df[cat].value_counts().reset_index()
        data.columns = [cat, "count"]
        fig = px.bar(data, x=cat, y="count", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 2. CLASSIFICATION ────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Classification")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xt)
    Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)

    algo = st.selectbox("Model", ["KNN","Decision Tree","Random Forest","GBRT"])
    if algo == "KNN":
        model = KNeighborsClassifier()
    elif algo == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algo == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()

    model.fit(Xt_s, yt)
    preds = model.predict(Xe_s)
    prob = model.predict_proba(Xe_s)[:,1]

    mets = {
        "Accuracy": accuracy_score(ye, preds),
        "Precision": precision_score(ye, preds),
        "Recall": recall_score(ye, preds),
        "F1": f1_score(ye, preds)
    }
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))

    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig = px.bar(imp, orientation='h', title="Feature Importance", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("What-If Analysis"):
        inp = pd.DataFrame([{
            c: st.slider(c, float(df[c].min()), float(df[c].max()), float(df[c].mean()))
            for c in X.columns
        }])
        p = model.predict(sc.transform(inp))[0]
        st.success("Will Try" if p == 1 else "Will Not Try")

    cm = confusion_matrix(ye, preds)
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred No","Pred Yes"],
        y=["Actual No","Actual Yes"],
        text=cm, texttemplate="%{text}",
        colorscale="Blues"
    ))
    fig_cm.update_layout(title="Confusion Matrix", template="plotly_white")
    st.plotly_chart(fig_cm, use_container_width=True)

    fpr, tpr, _ = roc_curve(ye, prob)
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', line_color="#EF476F"))
    fig_roc.update_layout(title="ROC Curve", template="plotly_white")
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 3. CLUSTERING ────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤝 Clustering")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]

    # Elbow
    inertias = [KMeans(n_clusters=k, random_state=42).fit(df[feats]).inertia_ for k in range(2,11)]
    fig_elbow = px.line(x=list(range(2,11)), y=inertias,
                        labels={'x':'K','y':'Inertia'},
                        markers=True, template="plotly_white")
    st.plotly_chart(fig_elbow, use_container_width=True)

    k = st.slider("Clusters (K)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(df[feats])
    df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig_centers = px.bar(centers, x=centers.index, y=feats,
                        barmode='group', template="plotly_white")
    st.plotly_chart(fig_centers, use_container_width=True)

    prof = df.groupby("Cluster")[feats].mean().round(1)
    st.subheader("Cluster Profiles")
    for c, row in prof.iterrows():
        st.info(f"Cluster {c}: " + ", ".join(f"{f}={v}" for f,v in row.items()))

    st.markdown('</div>', unsafe_allow_html=True)

# ── 4. ASSOCIATION ──────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔗 Association Rules")
    cols = [c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    support = st.slider("Min Support", 0.01, 0.2, 0.05)
    confidence = st.slider("Min Confidence", 0.1, 0.7, 0.3)
    freq = apriori(df[cols], min_support=support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=confidence)
    rules["rule"] = rules["antecedents"].apply(lambda x: ", ".join(x)) + " → " + rules["consequents"].apply(lambda x: ", ".join(x))
    top = rules.sort_values("lift", ascending=False).head(10)
    fig = px.bar(top, x="lift", y="rule", orientation='h',
                 color="confidence", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    G = nx.Graph()
    for _, r in top.iterrows():
        for a in r.antecedents:
            for b in r.consequents:
                G.add_edge(a, b)
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1), hoverinfo='none')
    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=list(G.nodes()), textposition='top center',
                            marker=dict(size=20, color="#EF476F"))
    fig_net = go.Figure([edge_trace, node_trace], layout=go.Layout(template="plotly_white", title="Rule Network"))
    st.plotly_chart(fig_net, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 5. ANOMALY ───────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚨 Anomaly Detection")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso = IsolationForest(contamination=0.05, random_state=42).fit(df[feats])
    df["Anomaly"] = iso.predict(df[feats])
    fig = px.scatter(df, x="MonthlyDisposableIncome", y="SpendPerServing",
                     color=df["Anomaly"].map({1:"Normal", -1:"Anomaly"}),
                     color_discrete_sequence=["#00FF00","#FF0000"], template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Outliers:** {(df['Anomaly']==-1).sum()}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 6. REGRESSION ────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Regression Comparison")
    Xr = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    yr = df["SpendPerServing"]
    Xt, Xe, yt, ye = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    models = {
        "Linear": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "Tree": DecisionTreeRegressor()
    }
    res = []
    for name, mdl in models.items():
        mdl.fit(Xt, yt)
        pr = mdl.predict(Xe)
        res.append({
            "Model": name,
            "R2": r2_score(ye, pr),
            "RMSE": np.sqrt(mean_squared_error(ye, pr))
        })
    rdf = pd.DataFrame(res)
    st.table(rdf)
    fig = px.bar(rdf, x="Model", y=["R2","RMSE"], barmode='group', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 7. FORECASTING ──────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⏱️ Forecasting")
    ts = df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    if len(ts) > 30:
        m = Prophet(); m.fit(ts)
        fut = m.make_future_dataframe(periods=30); fc = m.predict(fut)
        st.plotly_chart(plot_plotly(m, fc), use_container_width=True)
    else:
        st.warning("Not enough data for forecasting.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── 8. COHORT ───────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👥 Cohort Analysis")
    mode = st.radio("Cohort Type", ["Time","Cluster"])
    if mode == "Time":
        df["Month"] = df.SurveyDate.dt.to_period("M").astype(str)
        cr = df.groupby("Month")["SubscribePlan"].apply(lambda x: (x=="Yes").mean()).reset_index(name="Rate")
        fig = px.line(cr, x="Month", y="Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
        k = st.slider("Clusters", 2, 8, 4, key="cohort_k")
        cl = KMeans(n_clusters=k, random_state=42).fit_predict(df[feats])
        df["CohortCluster"] = cl
        cr = df.groupby("CohortCluster")["SubscribePlan"].apply(lambda x: (x=="Yes").mean()).reset_index(name="Rate")
        fig = px.bar(cr, x="CohortCluster", y="Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 9. GEOGRAPHY ────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺️ Geography")
    stats = df.groupby("City")["SpendPerServing"].mean().reset_index().sort_values("SpendPerServing", ascending=False)
    fig = px.bar(stats, x="City", y="SpendPerServing",
                 color="SpendPerServing", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 10. SENTIMENT ───────────────────────────────────────────────────────────
with tabs[9]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Sentiment Analysis")
    text = " ".join(df.Feedback.astype(str))
    wc = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig)
    sent = df.Feedback.astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    labels = pd.cut(sent, [-1,-0.1,0.1,1], labels=["Negative","Neutral","Positive"])
    cnt = labels.value_counts()
    fig2 = px.pie(names=cnt.index, values=cnt.values, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 11. LTV & CHURN ───────────────────────────────────────────────────────────
with tabs[10]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 LTV & Churn Prediction")
    df["FreqNum"] = df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"] = df.SpendPerServing * df.FreqNum * 12
    churn = (df.SubscribePlan=="No").astype(int)
    Xc = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt,Xe,yt,ye = train_test_split(Xc, churn, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42).fit(Xt, yt)
    preds = clf.predict(Xe)
    mets = {
        "Accuracy": accuracy_score(ye, preds),
        "Precision": precision_score(ye, preds),
        "Recall": recall_score(ye, preds)
    }
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))
    fig = px.histogram(df, x="LTV", nbins=30, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 12. PRICE ELASTICITY ─────────────────────────────────────────────────────
with tabs[11]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💵 Price Elasticity Simulator")
    price = st.slider("Price per Serving (AED)", 5, 30, 12)
    buyers = df[df.SpendPerServing >= price].shape[0]
    revenue = buyers * price
    st.metric("Expected Buyers", buyers)
    st.metric("Expected Revenue (AED)", revenue)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 13. GLOSSARY ────────────────────────────────────────────────────────────
with tabs[12]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📚 Glossary")
    glossary = {
        "SpendPerServing": "AED spent per serving.",
        "HealthConsciousness": "Self-rated health importance (1-10).",
        "TryNewBrand": "Willingness to try a new brand (Yes/No).",
        "SubscribePlan": "Subscription status (Yes/No).",
        "MonthlyDisposableIncome": "Monthly disposable income (AED).",
        "Age": "Age of respondent.",
        "ExerciseFrequency": "How often respondent exercises.",
        "ConsumptionFrequency": "How often health drink is consumed.",
        "Feedback": "Free-text user feedback."
    }
    for k, v in glossary.items():
        st.markdown(f"**{k}** <abbr title='{v}'>ℹ️</abbr>")
    st.markdown('</div>', unsafe_allow_html=True)
