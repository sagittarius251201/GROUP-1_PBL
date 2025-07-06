# app.py

# ── STREAMLIT DASHBOARD FOR HEALTH DRINK ANALYSIS ──

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

# --- THEME SETUP ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
theme = st.sidebar.radio("Theme", ["Light","Dark"], index=0)
st.session_state.theme = theme
if theme == "Dark":
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

# --- LOGO & DATA LOADING ---
logo = st.sidebar.file_uploader("Upload Logo", type=["png","jpg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

def load_data():
    data_path = Path(__file__).parent/"data"/"health_drink_survey_1000_augmented.csv"
    if data_path.exists():
        return pd.read_csv(data_path, parse_dates=["SurveyDate"])
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        return pd.read_csv(uploaded, parse_dates=["SurveyDate"])
    st.sidebar.error("No data file found.")
    st.stop()

if st.sidebar.button("Refresh Data"):
    df = load_data()
    st.sidebar.write(f"Refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

# --- SIDEBAR FILTERS ---
filters = []
def add_filter(label, selection):
    if isinstance(selection, list) and selection:
        filters.append(f"{label}: {', '.join(map(str,selection))}")

with st.sidebar.expander("Demographics", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    gender = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(gender)]; add_filter("Gender", gender)
    occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occ)]; add_filter("Occupation", occ)

with st.sidebar.expander("Behavior"):
    ex = st.multiselect("Exercise Frequency", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(ex)]; add_filter("ExerciseFrequency", ex)
    cf = st.multiselect("Consumption Frequency", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]; add_filter("ConsumptionFrequency", cf)

with st.sidebar.expander("Subscription & Location"):
    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]; add_filter("SubscribePlan", sub)
    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]; add_filter("City", city)

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")


if filters:
    chips = " ".join(
        f"<span style='background:{hl};color:#FFF;padding:4px 8px;border-radius:4px;margin:2px'>{f}</span>"
        for f in filters
    )
    st.markdown("**Active Filters:** " + chips, unsafe_allow_html=True)

# --- PAGE NAVIGATION ---
pages = [
    "Visualization","Classification","Clustering","Association","Anomaly","Regression",
    "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Chat","Glossary"
]
icons = [
    "bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
    "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","chat","book"
]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal",
                   styles={"nav-link-selected":{"background-color":hl}})
st.markdown(f"**Home** → **{page}**")

# --- 1. Visualization ---
if page == "Visualization":
    st.header("📊 Visualization")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Spend", f"AED {df.SpendPerServing.mean():.2f}")
    col2.metric("Trial Rate", f"{(df.TryNewBrand=='Yes').mean()*100:.1f}%")
    col3.metric("Subscribers", f"{(df.SubscribePlan=='Yes').sum()}")

    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()

    if chart == "Scatter":
        x = st.selectbox("X-axis", nums, 0)
        y = st.selectbox("Y-axis", nums, 1)
        fig = px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                         opacity=0.7, template=tpl)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Higher **{x}** tends to accompany higher **{y}**.")

    elif chart == "Histogram":
        col = st.selectbox("Feature", nums)
        fig = px.histogram(df, x=col, nbins=30, template=tpl)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Average **{col}** is {df[col].mean():.2f}.")

    elif chart == "Box":
        cat = st.selectbox("Category", cats)
        val = st.selectbox("Value", nums)
        fig = px.box(df, x=cat, y=val, template=tpl)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Distribution of **{val}** varies by **{cat}**.")

    else:
        cat = st.selectbox("Category", cats)
        metric = st.selectbox("Metric", ["Count","Mean Spend"])
        if metric == "Count":
            data = df[cat].value_counts().reset_index()
            data.columns = [cat, "count"]
            fig = px.bar(data, x=cat, y="count", template=tpl)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Insight:** Largest segment: **{data.iloc[0][cat]}**.")
        else:
            agg = df.groupby(cat)["SpendPerServing"].mean().reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", template=tpl)
            st.plotly_chart(fig, use_container_width=True)
            top_seg = agg.loc[agg.SpendPerServing.idxmax(), cat]
            st.markdown(f"**Insight:** Highest spend segment: **{top_seg}**.")

# --- 2. Classification ---
elif page == "Classification":
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
    preds = model.predict(Xe_s)
    prob = model.predict_proba(Xe_s)[:,1]
    mets = {
        "Accuracy":accuracy_score(ye, preds),
        "Precision":precision_score(ye, preds),
        "Recall":recall_score(ye, preds),
        "F1":f1_score(ye, preds)
    }
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))

    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig = px.bar(imp, orientation='h', title="Feature Importance", template=tpl)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("What-If Analysis"):
        inp = pd.DataFrame([{c: st.slider(c, float(df[c].min()), float(df[c].max()), float(df[c].mean())) for c in X.columns}])
        p = model.predict(sc.transform(inp))[0]
        st.success("Will Try" if p==1 else "Will Not Try")

    cm = confusion_matrix(ye, preds)
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Pred No","Pred Yes"], y=["Actual No","Actual Yes"],
        text=cm, texttemplate="%{text}", colorscale="Blues"
    ))
    fig_cm.update_layout(title="Confusion Matrix", template=tpl)
    st.plotly_chart(fig_cm, use_container_width=True)

    fpr, tpr, _ = roc_curve(ye, prob)
    fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', line_color=hl))
    fig_roc.update_layout(title="ROC Curve", template=tpl)
    st.plotly_chart(fig_roc, use_container_width=True)

# --- 3. Clustering ---
elif page == "Clustering":
    st.header("🤝 Clustering")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    st.subheader("Elbow Method")
    inertias = [KMeans(n_clusters=k, random_state=42).fit(df[feats]).inertia_ for k in range(2,11)]
    fig_elbow = px.line(x=list(range(2,11)), y=inertias, markers=True, labels={'x':'K','y':'Inertia'}, template=tpl)
    st.plotly_chart(fig_elbow, use_container_width=True)
    k = st.slider("Clusters (K)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(df[feats])
    df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig_centers = px.bar(centers, x=centers.index, y=feats, barmode='group', template=tpl)
    st.plotly_chart(fig_centers, use_container_width=True)

    prof = df.groupby("Cluster")[feats].mean().round(1)
    st.subheader("Cluster Profiles")
    for c, row in prof.iterrows():
        st.info(f"Cluster {c}: " + ", ".join(f"{f}={v}" for f,v in row.items()))

    st.subheader("Compare Clusters")
    a = st.selectbox("Cluster A", prof.index)
    b = st.selectbox("Cluster B", prof.index, key="cmp")
    diff = (prof.loc[a] - prof.loc[b]).to_frame("Difference").T
    st.table(diff)

# ... continue other pages similarly ...
# --- CHAT PAGE (at end) ---
elif page == "Chat":
    st.header("💬 Ask GPT")
    try:
        import openai
    except ImportError:
        st.warning("ChatGPT feature unavailable. Ensure `openai` is in requirements.txt.")
    else:
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

# --- GLOSSARY ---
elif page == "Glossary":
    st.header("📚 Glossary")
    glossary = {
        "SpendPerServing": "AED spent per serving.",
        "HealthConsciousness": "Health importance (1–10).",
        "TryNewBrand": "Willingness to try new brand (Yes/No).",
        "SubscribePlan": "Subscription status.",
        "MonthlyDisposableIncome": "Disposable income per month.",
        "Age": "Respondent age.",
        "ExerciseFrequency": "Exercise frequency.",
        "ConsumptionFrequency": "Consumption frequency.",
        "Feedback": "User comments."
    }
    for k,v in glossary.items():
        st.markdown(f"**{k}**: {v}")

# --- FEEDBACK BUTTON ---
if st.button("💡 Feedback"):
    fb = st.text_area("Your feedback:")
    if st.button("Submit"):
        with open("feedback.txt","a") as f:
            f.write(fb + "\n---\n")
        st.success("Thank you!")
