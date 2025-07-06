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
from textblob import TextBlob
import networkx as nx
import time

# Try importing OpenAI, but don’t crash if it’s missing
try:
    import openai
    from openai.error import RateLimitError, OpenAIError
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False

# --- Page & Theme Setup ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")

theme = st.sidebar.radio("Theme", ['Light','Dark'], index=0)
st.session_state.theme = theme
if theme == 'Dark':
    bg, fg, sbg, hl, tpl = "#1B1B2F","#FFFFFF","#30475E","#EF476F","plotly_dark"
else:
    bg, fg, sbg, hl, tpl = "#FFFFFF","#000000","#F8F9FB","#EF476F","plotly_white"
st.markdown(f"""
    <style>
    body {{background-color:{bg}; color:{fg};}}
    .sidebar .sidebar-content {{background-color:{sbg};}}
    h1,h2,h3,h4 {{color:{hl};}}
    </style>
""", unsafe_allow_html=True)

# --- Logo & Data Loader ---
logo = st.sidebar.file_uploader("Upload Logo", type=["png","jpg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

def load_data():
    path = Path(__file__).parent/"data"/"health_drink_survey_1000_augmented.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["SurveyDate"])
    up = st.sidebar.file_uploader("Upload CSV", type="csv")
    if up:
        return pd.read_csv(up, parse_dates=["SurveyDate"])
    st.sidebar.error("No data found."); st.stop()

if st.sidebar.button("Refresh Data"):
    df = load_data()
    st.sidebar.write(f"Refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

# --- Sidebar Filters ---
filters = []
def safe_join(var, label):
    if isinstance(var, list) and var:
        filters.append(f"{label}: " + ", ".join(map(str, var)))

with st.sidebar.expander("Demographics", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(g)]; safe_join(g, "Gender")
    occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occ)]; safe_join(occ, "Occ")

with st.sidebar.expander("Behavior"):
    ex = st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(ex)]; safe_join(ex, "ExFreq")
    cf = st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]; safe_join(cf, "ConFreq")

with st.sidebar.expander("Subscription & Location"):
    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]; safe_join(sub, "SubPlan")
    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]; safe_join(city, "City")

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("Download Filtered CSV", df.to_csv(index=False).encode(), "filtered.csv")
if filters:
    chips = " ".join(f"<span style='background:{hl};color:#FFF;padding:3px 8px;border-radius:4px;margin:2px'>{f}</span>" for f in filters)
    st.markdown("**Active Filters:** " + chips, unsafe_allow_html=True)

# --- Navigation ---
pages = [
    "Visualization","Classification","Clustering","Association","Anomaly","Regression",
    "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity",
    "Chat","Glossary"
]
icons = [
    "bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
    "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin",
    "chat","book"
]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal",
                   styles={"nav-link-selected":{"background-color":hl}})
st.markdown(f"**Home** → **{page}**")

# --- Page Implementations ---

if page=="Visualization":
    st.header("📊 Visualization")
    # ... your visualization code here, use template=tpl on all Plotly charts ...

elif page=="Classification":
    st.header("🤖 Classification")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xt)
    Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)
    from sklearn.neighbors import KNeighborsClassifier
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (
        KNeighborsClassifier() if algo=="KNN" else
        DecisionTreeClassifier() if algo=="Decision Tree" else
        RandomForestClassifier() if algo=="Random Forest" else
        GradientBoostingClassifier()
    )
    model.fit(Xt_s, yt)
    p = model.predict(Xe_s)
    prob = model.predict_proba(Xe_s)[:,1]
    mets = {
        "Accuracy":accuracy_score(ye, p),
        "Precision":precision_score(ye, p),
        "Recall":recall_score(ye, p),
        "F1":f1_score(ye, p)
    }
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))

    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig = px.bar(imp, orientation='h', title="Feature Importance",
                     color=imp, color_continuous_scale='Viridis',
                     labels={'value':'Importance','index':'Feature'},
                     template=tpl)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("What-If Analysis"):
        row = {c: st.slider(c, float(df[c].min()), float(df[c].max()), float(df[c].mean())) for c in X.columns}
        inp = pd.DataFrame([row])
        pred = model.predict(sc.transform(inp))[0]
        st.success("Will Try" if pred==1 else "Will Not Try")

    cm = confusion_matrix(ye, p)
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Pred No","Pred Yes"], y=["Actual No","Actual Yes"],
        text=cm, texttemplate="%{text}", colorscale="Blues"
    ))
    fig_cm.update_layout(title="Confusion Matrix", template=tpl)
    st.plotly_chart(fig_cm, use_container_width=True)

    fpr, tpr, _ = roc_curve(ye, prob)
    fig2 = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', line_color=hl))
    fig2.update_layout(title="ROC Curve", template=tpl)
    st.plotly_chart(fig2, use_container_width=True)

elif page=="Clustering":
    st.header("🤝 Clustering")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    st.subheader("Elbow Method")
    inertias = [KMeans(n_clusters=k, random_state=42).fit(df[feats]).inertia_ for k in range(2,11)]
    fig = px.line(x=list(range(2,11)), y=inertias, markers=True,
                  labels={'x':'K','y':'Inertia'}, template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Choose K at the elbow point.")

    k = st.slider("Clusters (K)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(df[feats])
    df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig2 = px.bar(centers, x=centers.index, y=feats, barmode='group', template=tpl)
    st.plotly_chart(fig2, use_container_width=True)

    prof = df.groupby("Cluster")[feats].mean().round(1)
    st.subheader("Cluster Profiles")
    for c, row in prof.iterrows():
        st.info(f"Cluster {c}: " + ", ".join(f"{f}={v}" for f,v in row.items()))

    st.subheader("Compare Clusters")
    a = st.selectbox("Cluster A", prof.index)
    b = st.selectbox("Cluster B", prof.index, key="cmp")
    diff = (prof.loc[a] - prof.loc[b]).to_frame("Difference").T
    st.table(diff)

elif page=="Association":
    st.header("🔗 Association Rules")
    cols = [c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    s = st.slider("Min Support", 0.01, 0.2, 0.05)
    cth = st.slider("Min Confidence", 0.1, 0.7, 0.3)
    freq = apriori(df[cols], min_support=s, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=cth)
    rules["rule"] = rules["antecedents"].apply(lambda x:", ".join(x)) + " → " + rules["consequents"].apply(lambda x:", ".join(x))
    top = rules.sort_values("lift", ascending=False).head(10)
    fig = px.bar(top, x="lift", y="rule", orientation='h',
                 color="confidence", color_continuous_scale="Plasma", template=tpl)
    st.plotly_chart(fig, use_container_width=True)

    # Network graph
    G = nx.Graph()
    for _, r in top.iterrows():
        for a in r.antecedents:
            for b in r.consequents:
                G.add_edge(a, b)
    pos = nx.spring_layout(G)
    edge_x = []; edge_y = []
    for e in G.edges():
        x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1), hoverinfo='none')
    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=list(G.nodes()), textposition='top center',
                            marker=dict(size=20, color=hl))
    fig_net = go.Figure([edge_trace, node_trace], layout=go.Layout(template=tpl, title="Rule Network"))
    st.plotly_chart(fig_net, use_container_width=True)

elif page=="Anomaly":
    st.header("🚨 Anomaly Detection")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso = IsolationForest(contamination=0.05, random_state=42).fit(df[feats])
    df["Anomaly"] = iso.predict(df[feats])
    fig = px.scatter(df, x="MonthlyDisposableIncome", y="SpendPerServing",
                     color=df["Anomaly"].map({1:"Normal", -1:"Anomaly"}),
                     color_discrete_sequence=["#00FF00","#FF0000"], template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Outliers:** {(df['Anomaly']==-1).sum()}")

elif page=="Regression":
    st.header("📈 Regression Comparison")
    Xr = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    yr = df["SpendPerServing"]
    Xt, Xe, yt, ye = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    models = {
        "Linear":LinearRegression(),
        "Lasso":Lasso(),
        "Ridge":Ridge(),
        "Tree":DecisionTreeRegressor()
    }
    res = []
    for name, mdl in models.items():
        mdl.fit(Xt, yt)
        pr = mdl.predict(Xe)
        res.append({"Model":name, "R2":r2_score(ye,pr), "RMSE":np.sqrt(mean_squared_error(ye,pr))})
    rdf = pd.DataFrame(res)
    st.table(rdf)
    fig = px.bar(rdf, x="Model", y=["R2","RMSE"], barmode='group', template=tpl)
    st.plotly_chart(fig, use_container_width=True)

elif page=="Forecasting":
    st.header("⏱️ Forecasting")
    ts = df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    if len(ts) > 30:
        m = Prophet(); m.fit(ts)
        fut = m.make_future_dataframe(periods=30)
        fc = m.predict(fut)
        st.plotly_chart(plot_plotly(m, fc), use_container_width=True)
    else:
        st.warning("Not enough data for forecasting.")

elif page=="Cohort":
    st.header("👥 Cohort Analysis")
    mode = st.radio("Cohort Type", ["Time","Cluster"])
    if mode=="Time":
        df["Month"] = df.SurveyDate.dt.to_period("M").astype(str)
        cr = df.groupby("Month")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.line(cr, x="Month", y="Rate", template=tpl); st.plotly_chart(fig, use_container_width=True)
    else:
        if "Cluster" not in df:
            st.error("Run Clustering first"); st.stop()
        cr = df.groupby("Cluster")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.bar(cr, x="Cluster", y="Rate", template=tpl); st.plotly_chart(fig, use_container_width=True)

elif page=="Geography":
    st.header("🗺️ Geography")
    city_stats = df.groupby("City")["SpendPerServing"].mean().reset_index().sort_values("SpendPerServing", ascending=False)
    fig = px.bar(city_stats, x="City", y="SpendPerServing", color="SpendPerServing",
                 color_continuous_scale="Viridis", template=tpl)
    st.plotly_chart(fig, use_container_width=True)

elif page=="Sentiment":
    st.header("💬 Sentiment")
    text = " ".join(df.Feedback.astype(str))
    wc = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig)
    sent = df.Feedback.astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    labels = pd.cut(sent, [-1,-0.1,0.1,1], labels=["Negative","Neutral","Positive"])
    cnt = labels.value_counts()
    fig2 = px.pie(names=cnt.index, values=cnt.values, template=tpl); st.plotly_chart(fig2, use_container_width=True)

elif page=="LTV & Churn":
    st.header("💰 LTV & Churn")
    df["FreqNum"] = df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"] = df.SpendPerServing * df.FreqNum * 12
    churn = (df.SubscribePlan=="No").astype(int)
    Xc = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt, Xe, yt, ye = train_test_split(Xc, churn, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42).fit(Xt, yt)
    pr = clf.predict(Xe)
    mets = {"Accuracy":accuracy_score(ye,pr),
            "Precision":precision_score(ye,pr),
            "Recall":recall_score(ye,pr)}
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))
    fig = px.histogram(df, x="LTV", nbins=30, template=tpl); st.plotly_chart(fig, use_container_width=True)
    churn_prob = clf.predict_proba(Xe)[:,1]
    fig2 = px.histogram(churn_prob, nbins=20, title="Churn Probability", template=tpl)
    st.plotly_chart(fig2, use_container_width=True)

elif page=="Price Elasticity":
    st.header("💵 Price Elasticity")
    price = st.slider("Price per Serving",5,30,12)
    buyers = df[df.SpendPerServing>=price].shape[0]
    rev = buyers * price
    st.metric("Buyers", buyers); st.metric("Revenue AED", rev)

elif page=="Chat":
    st.header("💬 Ask GPT")
    if not CHAT_AVAILABLE:
        st.warning("ChatGPT feature unavailable. Please add `openai` to requirements.")
    else:
        if "msgs" not in st.session_state:
            st.session_state.msgs = [{"role":"system","content":"You are a data analyst assistant."}]
        query = st.chat_input("Your question...")
        if query:
            st.session_state.msgs.append({"role":"user","content":query})
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            try:
                resp = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.msgs
                )
                msg = resp["choices"][0]["message"]
                st.session_state.msgs.append({"role":msg["role"],"content":msg["content"]})
            except RateLimitError:
                st.error("Rate limit exceeded; try again in a minute.")
            except OpenAIError as e:
                st.error(f"OpenAI API error: {e}")
        for m in st.session_state.msgs:
            st.chat_message(m["role"]).write(m["content"])

elif page=="Glossary":
    st.header("📚 Glossary")
    gloss = {
        "SpendPerServing":"AED spent per serving.",
        "HealthConsciousness":"Health importance (1–10).",
        "TryNewBrand":"Will try new brand (Yes/No).",
        "SubscribePlan":"Subscription status.",
        "MonthlyDisposableIncome":"Disposable income/month.",
        "Age":"Respondent age.",
        "ExerciseFrequency":"Exercise frequency.",
        "ConsumptionFrequency":"Consumption frequency.",
        "Feedback":"User comments."
    }
    for k,v in gloss.items():
        st.markdown(f"**{k}**: {v}")

if st.button("💡 Feedback"):
    fb = st.text_area("Your feedback:")
    if st.button("Submit"):
        with open("feedback.txt","a") as f:
            f.write(fb + "\n---\n")
        st.success("Thanks!")    
