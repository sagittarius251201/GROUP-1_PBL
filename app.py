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

# ─────────────────────────────────────────────────────────────────────────────
# Page config & custom CSS for cards & tooltips
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Health Drink Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    margin: 10px 0;
}
.dark .card {
    background-color: #2E2E2E;
}
.tooltip {
    border-bottom: 1px dotted #666;
    cursor: help;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Theming logic
# ─────────────────────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
st.session_state.theme = theme

if theme == "Dark":
    bg, fg, sbg, hl, tpl = "#1B1B2F","#FFFFFF","#30475E","#EF476F","plotly_dark"
else:
    bg, fg, sbg, hl, tpl = "#FFFFFF","#000000","#F8F9FB","#EF476F","plotly_white"

st.markdown(f"""
    <style>
    body {{ background-color: {bg}; color: {fg}; }}
    .sidebar .sidebar-content {{ background-color: {sbg}; }}
    h1,h2,h3,h4 {{ color: {hl}; }}
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    path = Path(__file__).parent/"data"/"health_drink_survey_1000_augmented.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["SurveyDate"])
    uploaded = st.sidebar.file_uploader("Upload data CSV", type="csv")
    if uploaded:
        return pd.read_csv(uploaded, parse_dates=["SurveyDate"])
    st.sidebar.error("No data found.")
    st.stop()

if st.sidebar.button("🔄 Refresh Data"):
    df = load_data()
    st.sidebar.success(f"Data refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────────────────────────────────────
filters = []
def add_filter(label, sel):
    if sel:
        if isinstance(sel, list):
            filters.append(f"{label}: {', '.join(map(str,sel))}")
        else:
            filters.append(f"{label}: {sel}")

with st.sidebar.expander("Demographics", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        add_filter("Age", f"{age[0]}–{age[1]}")
    gender = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(gender)]; add_filter("Gender", gender)
    occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occ)]; add_filter("Occupation", occ)

with st.sidebar.expander("Behavior"):
    ex = st.multiselect("Exercise Frequency", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(ex)]; add_filter("ExerciseFreq", ex)
    cf = st.multiselect("Consumption Frequency", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]; add_filter("ConsumptionFreq", cf)

with st.sidebar.expander("Subscription & Location"):
    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]; add_filter("SubscribePlan", sub)
    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]; add_filter("City", city)

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(dr[0], dr[1])]
        add_filter("Date", f"{dr[0]}–{dr[1]}")

st.sidebar.download_button("📥 Download Filtered CSV", df.to_csv(index=False).encode(), "filtered.csv")

if filters:
    chips = " ".join(f"<span class='card tooltip' title='{f}'>{f}</span>" for f in filters)
    st.markdown(f"**Active Filters:**  {chips}", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Navigation menu
# ─────────────────────────────────────────────────────────────────────────────
pages = [
    "Visualization","Classification","Clustering","Association","Anomaly","Regression",
    "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Chat","Glossary"
]
icons = [
    "bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
    "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","chat","book"
]
page = option_menu(None, pages, icons=icons, menu_icon="palette", orientation="horizontal",
                   styles={"nav-link-selected":{"background-color":hl}})
st.markdown(f"**Home** → **{page}**")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Visualization
# ─────────────────────────────────────────────────────────────────────────────
if page == "Visualization":
    st.header("📊 Data Visualization")
    # Metrics cards
    avg_spend = df.SpendPerServing.mean()
    trial_rate = (df.TryNewBrand=='Yes').mean()*100
    subs = (df.SubscribePlan=='Yes').sum()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='card'><h3>Avg Spend <span class='tooltip' title='Average amount spent per serving'>❔</span></h3><h2>AED {avg_spend:.2f}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h3>Trial Rate <span class='tooltip' title='% willing to try new brand'>❔</span></h3><h2>{trial_rate:.1f}%</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><h3>Subscribers <span class='tooltip' title='Number of subscribers'>❔</span></h3><h2>{subs}</h2></div>", unsafe_allow_html=True)

    # Chart selector
    chart = st.selectbox("Choose Chart Type", ["Scatter","Histogram","Box","Bar"], key="vis_chart")
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()

    if chart == "Scatter":
        x = st.selectbox("X-axis", nums, 0)
        y = st.selectbox("Y-axis", nums, 1)
        fig = px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                         opacity=0.7, template=tpl, color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Strong correlation between **{x}** and **{y}**, ideal for targeting.")
    elif chart == "Histogram":
        col = st.selectbox("Feature", nums, key="hist_col")
        fig = px.histogram(df, x=col, nbins=30, template=tpl, color_discrete_sequence=["#EF476F"])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Mean **{col}** ≈ {df[col].mean():.2f}.")
    elif chart == "Box":
        cat = st.selectbox("Category", cats, key="box_cat")
        val = st.selectbox("Value", nums, key="box_val")
        fig = px.box(df, x=cat, y=val, template=tpl, color_discrete_sequence=["#06D6A0"])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** **{val}** varies significantly across **{cat}**.")
    else:
        cat = st.selectbox("Category", cats, key="bar_cat")
        metric = st.selectbox("Metric", ["Count","Mean Spend"], key="bar_metric")
        if metric == "Count":
            data = df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig = px.bar(data, x=cat, y="count", template=tpl, color_discrete_sequence=["#FFD166"])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Insight:** Top segment is **{data.iloc[0][cat]}**.")
        else:
            agg = df.groupby(cat)["SpendPerServing"].mean().reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", template=tpl, color_discrete_sequence=["#EF476F"])
            st.plotly_chart(fig, use_container_width=True)
            top = agg.loc[agg.SpendPerServing.idxmax(), cat]
            st.markdown(f"**Insight:** **{top}** spends most on average.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Classification
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Classification":
    st.header("🤖 Classification Models")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xt)
    Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)

    algo = st.selectbox("Select Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    if algo == "KNN":
        model = __import__('sklearn.neighbors', fromlist=['KNeighborsClassifier']).KNeighborsClassifier()
    elif algo == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algo == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()
    model.fit(Xt_s, yt)
    preds = model.predict(Xe_s)
    prob = model.predict_proba(Xe_s)[:,1]

    metrics = {
        "Accuracy": accuracy_score(ye, preds),
        "Precision": precision_score(ye, preds),
        "Recall": recall_score(ye, preds),
        "F1-score": f1_score(ye, preds)
    }
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig = px.bar(imp, orientation='h', title="Feature Importances", template=tpl,
                     color=imp, color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Most important: **{imp.idxmax()}**.")

    # Confusion matrix
    cm = confusion_matrix(ye, preds)
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Pred No","Pred Yes"], y=["Act No","Act Yes"],
        colorscale="Blues", text=cm, texttemplate="%{text}"
    ))
    fig_cm.update_layout(title="Confusion Matrix", template=tpl)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(ye, prob)
    fig_roc = px.area(
        x=fpr, y=tpr,
        title='ROC Curve',
        labels={'x':'False Positive Rate', 'y':'True Positive Rate'},
        template=tpl
    )
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)

    # What-if explainer
    with st.expander("🔮 What-If Analysis"):
        inp = pd.DataFrame([{c: st.slider(c, float(df[c].min()), float(df[c].max()), float(df[c].mean())) for c in X.columns}])
        pred = model.predict(sc.transform(inp))[0]
        st.success("✅ Will Try" if pred==1 else "❌ Will Not Try")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Clustering
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Clustering":
    st.header("🤝 Customer Segmentation")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    st.subheader("Elbow Method")
    inertias = [KMeans(n_clusters=k, random_state=42).fit(df[feats]).inertia_ for k in range(2,11)]
    fig_elbow = px.line(x=list(range(2,11)), y=inertias, markers=True,
                        labels={'x':'#Clusters','y':'Inertia'}, template=tpl)
    st.plotly_chart(fig_elbow, use_container_width=True)

    k = st.slider("Number of Clusters", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(df[feats])
    df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig_centers = px.bar(centers, x=centers.index, y=feats, barmode='group', template=tpl)
    st.plotly_chart(fig_centers, use_container_width=True)

    st.subheader("Cluster Profiles")
    prof = df.groupby("Cluster")[feats].mean().round(1)
    for c, row in prof.iterrows():
        st.markdown(f"<div class='card'><b>Cluster {c}</b>: " +
                    ", ".join(f"{f}={v}" for f,v in row.items()) +
                    "</div>", unsafe_allow_html=True)

    st.subheader("Compare Clusters")
    c1, c2 = st.selectbox("Cluster A", prof.index), st.selectbox("Cluster B", prof.index, key="cmp")
    diff = (prof.loc[c1] - prof.loc[c2]).to_frame("Difference").T
    st.table(diff)
    st.download_button("📥 Download Clusters", df.to_csv(index=False).encode(), "clusters.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Association Rule Mining
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Association":
    st.header("🔗 Association Rules")
    cols = [c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    s = st.slider("Min Support", 0.01, 0.2, 0.05)
    cth = st.slider("Min Confidence", 0.1, 0.7, 0.3)
    freq = apriori(df[cols], min_support=s, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=cth)
    rules["rule"] = rules.apply(lambda r: 
        ", ".join(r.antecedents) + " → " + ", ".join(r.consequents), axis=1
    )
    top = rules.sort_values("lift", ascending=False).head(10)
    fig = px.bar(top, x="lift", y="rule", orientation='h',
                 color="confidence", template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Insight:** Strongest rule: **{top.iloc[0]['rule']}**")

    st.subheader("Network Graph")
    G = nx.Graph()
    for _, r in top.iterrows():
        for a in r.antecedents:
            for b in r.consequents:
                G.add_edge(a, b, weight=r.lift)
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1))
    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
        textposition='top center', marker=dict(size=20, color=hl)
    )
    fig_net = go.Figure([edge_trace, node_trace], layout=go.Layout(template=tpl))
    st.plotly_chart(fig_net, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Anomaly":
    st.header("🚨 Anomaly Detection")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso = IsolationForest(contamination=0.05, random_state=42).fit(df[feats])
    df["Anomaly"] = iso.predict(df[feats])
    fig = px.scatter(
        df, x="MonthlyDisposableIncome", y="SpendPerServing",
        color=df["Anomaly"].map({1:"Normal",-1:"Anomaly"}),
        color_discrete_sequence=["#06D6A0","#EF476F"], template=tpl
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Outliers detected:** {(df.Anomaly==-1).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Regression Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Regression":
    st.header("📈 Regression Models")
    Xr = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    yr = df["SpendPerServing"]
    Xt, Xe, yt, ye = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    models = {
        "Linear": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "Tree": DecisionTreeRegressor()
    }
    results = []
    for name, mdl in models.items():
        mdl.fit(Xt, yt)
        pr = mdl.predict(Xe)
        results.append({
            "Model": name,
            "R2": r2_score(ye, pr),
            "RMSE": np.sqrt(mean_squared_error(ye, pr))
        })
    res_df = pd.DataFrame(results)
    st.table(res_df)
    fig = px.bar(res_df, x="Model", y=["R2","RMSE"], barmode='group', template=tpl)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Forecasting
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Forecasting":
    st.header("⏱️ Time Series Forecast")
    ts = df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    if len(ts) > 30:
        m = Prophet(); m.fit(ts)
        fut = m.make_future_dataframe(periods=30); fc = m.predict(fut)
        st.plotly_chart(plot_plotly(m, fc), use_container_width=True)
        next_avg = fc.iloc[-7:]['yhat'].mean()
        st.markdown(f"**Insight:** Next week avg spend: AED {next_avg:.2f}")
    else:
        st.warning("Not enough data to forecast.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Cohort Analysis (Time & Cluster)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Cohort":
    st.header("👥 Cohort Analysis")
    mode = st.radio("Cohort by:", ["Time","Cluster"])
    if mode == "Time":
        df["Month"] = df.SurveyDate.dt.to_period("M").astype(str)
        cr = df.groupby("Month")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.line(cr, x="Month", y="Rate", markers=True, template=tpl)
        st.plotly_chart(fig, use_container_width=True)
    else:
        feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
        k = st.slider("Clusters for Cohort", 2, 8, 4)
        cl = KMeans(n_clusters=k, random_state=42).fit_predict(df[feats])
        df["CohortCluster"] = cl
        cr = df.groupby("CohortCluster")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.bar(cr, x="CohortCluster", y="Rate", template=tpl)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Geography
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Geography":
    st.header("🗺️ Geographic Insights")
    city_stats = df.groupby("City")["SpendPerServing"].mean().reset_index().sort_values("SpendPerServing", ascending=False)
    fig = px.bar(city_stats, x="City", y="SpendPerServing", color="SpendPerServing",
                 color_continuous_scale="Viridis", template=tpl)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Sentiment":
    st.header("💬 Sentiment Word Cloud & Pie")
    text = " ".join(df.Feedback.astype(str))
    wc = WordCloud(width=800, height=400, background_color="#FFF").generate(text)
    fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig)

    sent = df.Feedback.astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    labels = pd.cut(sent, [-1,-0.1,0.1,1], labels=["Negative","Neutral","Positive"])
    cnt = labels.value_counts()
    fig2 = px.pie(names=cnt.index, values=cnt.values, template=tpl)
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 11. LTV & Churn
# ─────────────────────────────────────────────────────────────────────────────
elif page == "LTV & Churn":
    st.header("💰 LTV & Churn Risk")
    df["FreqNum"] = df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"] = df.SpendPerServing * df.FreqNum * 12
    churn = (df.SubscribePlan=="No").astype(int)
    Xc = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt, Xe, yt, ye = train_test_split(Xc, churn, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42).fit(Xt, yt)
    preds = clf.predict(Xe)
    metrics = {
        "Accuracy": accuracy_score(ye, preds),
        "Precision": precision_score(ye, preds),
        "Recall": recall_score(ye, preds)
    }
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    st.plotly_chart(px.histogram(df, x="LTV", nbins=30, template=tpl), use_container_width=True)
    prob = clf.predict_proba(Xe)[:,1]
    st.plotly_chart(px.histogram(prob, nbins=20, title="Churn Probability", template=tpl), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 12. Price Elasticity
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Price Elasticity":
    st.header("💵 Price Elasticity Simulator")
    price = st.slider("Price per Serving (AED)", 5, 30, 12)
    buyers = df[df.SpendPerServing >= price].shape[0]
    revenue = buyers * price
    st.metric("Projected Buyers", buyers, delta=None, label_visibility="visible")
    st.metric("Projected Revenue (AED)", revenue, delta=None, label_visibility="visible")

# ─────────────────────────────────────────────────────────────────────────────
# 13. ChatGPT Q&A
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Chat":
    st.header("💬 Ask DataGPT")
    try:
        import openai
    except ImportError:
        st.warning("ChatGPT unavailable. Add `openai` to requirements.txt.")
    else:
        if "msgs" not in st.session_state:
            st.session_state.msgs = [{"role":"system","content":"You are a data analyst assistant."}]
        query = st.chat_input("Ask about the data or code...")
        if query:
            st.session_state.msgs.append({"role":"user","content":query})
            openai.api_key = st.secrets.get("OPENAI_API_KEY","")
            # trim history
            history = st.session_state.msgs[-6:]
            try:
                resp = openai.chat.completions.create(model="gpt-3.5-turbo", messages=history)
                msg = resp.choices[0].message
                st.session_state.msgs.append({"role":msg.role,"content":msg.content})
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
        for m in st.session_state.msgs:
            st.chat_message(m["role"]).write(m["content"])

# ─────────────────────────────────────────────────────────────────────────────
# 14. Glossary
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Glossary":
    st.header("📚 Data Glossary")
    glossary = {
        "SpendPerServing": "AED spent per serving",
        "HealthConsciousness": "Health importance (1–10)",
        "TryNewBrand": "Willingness to try new brand",
        "SubscribePlan": "Subscription status",
        "MonthlyDisposableIncome": "Disposable income/month",
        "Age": "Respondent age",
        "ExerciseFrequency": "Exercise frequency",
        "ConsumptionFrequency": "Consumption frequency",
        "Feedback": "Free-text user feedback"
    }
    for term, desc in glossary.items():
        st.markdown(f"**{term}**: {desc}")

# ─────────────────────────────────────────────────────────────────────────────
# Feedback
# ─────────────────────────────────────────────────────────────────────────────
if st.button("💡 Send Feedback"):
    fb = st.text_area("Your feedback:")
    if st.button("Submit Feedback"):
        with open("feedback.txt","a") as f:
            f.write(fb + "\n---\n")
        st.success("Thank you for your feedback!")
