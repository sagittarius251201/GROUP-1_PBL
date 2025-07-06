
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
import io
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# PDF generation with fallback
def generate_pdf(insights, figs):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Health Drink Market Analysis", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 24))
    for title, text in insights.items():
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 12))
    for idx, (caption, fig) in enumerate(figs.items(), start=1):
        story.append(Paragraph(f"<b>Figure {idx}: {caption}</b>", styles["Heading3"]))
        try:
            img_bytes = fig.to_image(format="png", width=600, height=400)
            img_buf = io.BytesIO(img_bytes)
            story.append(RLImage(img_buf, width=400, height=300))
        except Exception as e:
            story.append(Paragraph(f"*Could not export figure: {e}*", styles["BodyText"]))
        story.append(Spacer(1, 24))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# App configuration and CSS
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.markdown("""
<style>
.card { padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Theme toggle
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
is_dark = theme == "Dark"
tpl = "plotly_dark" if is_dark else "plotly_white"
st.markdown(f"<body class='{'dark' if is_dark else 'light'}'></body>", unsafe_allow_html=True)

# Load data
def load_data():
    path = Path(__file__).parent/"data"/"health_drink_survey_1000_augmented.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["SurveyDate"])
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        return pd.read_csv(uploaded, parse_dates=["SurveyDate"])
    st.error("No data found.")
    st.stop()

if st.sidebar.button("🔄 Refresh Data"):
    df = load_data()
    st.experimental_rerun()

df = load_data()

# Sidebar filters (no active filter chips)
with st.sidebar.expander("Filters", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
    gender = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(gender)]
    occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occ)]
    ex = st.multiselect("Exercise Frequency", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(ex)]
    cf = st.multiselect("Consumption Frequency", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]
    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]
    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min().date(), df.SurveyDate.max().date()])
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        df = df[(df.SurveyDate >= start) & (df.SurveyDate <= end)]
    st.sidebar.download_button("📥 Download Filtered CSV", df.to_csv(index=False).encode(), "filtered.csv")

# Navigation menu
pages = ["Visualization","Classification","Clustering","Association","Anomaly","Regression",
         "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Chat","Glossary"]
icons = ["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up","clock-history",
         "calendar","geo","chat-dots","cash-stack","cash-coin","chat","book"]
page = option_menu(None, pages, icons=icons, menu_icon="chart-bar", orientation="horizontal")
st.markdown(f"**Home** → **{page}**")

# Prepare per-page PDF data
pdf_insights = {}
pdf_figs = {}

# Visualization
if page == "Visualization":
    st.header("📊 Visualization")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Spend", f"AED {df.SpendPerServing.mean():.2f}")
    c2.metric("Trial Rate", f"{(df.TryNewBrand=='Yes').mean()*100:.1f}%")
    c3.metric("Subscribers", f"{(df.SubscribePlan=='Yes').sum()}")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if chart == "Scatter":
        x = st.selectbox("X-axis", nums, 0)
        y = st.selectbox("Y-axis", nums, 1)
        fig = px.scatter(df, x=x, y=y, color=cats[0] if cats else None, template=tpl)
        insight = f"Correlation between {x} and {y} suggests targeting."
    elif chart == "Histogram":
        col = st.selectbox("Feature", nums)
        fig = px.histogram(df, x=col, template=tpl)
        insight = f"Mean {col} ≈ {df[col].mean():.2f}."
    elif chart == "Box":
        cat = st.selectbox("Category", cats)
        val = st.selectbox("Value", nums)
        fig = px.box(df, x=cat, y=val, template=tpl)
        insight = f"Variation of {val} across {cat}."
    else:
        cat = st.selectbox("Category", cats)
        metric = st.selectbox("Metric", ["Count","Mean Spend"])
        if metric == "Count":
            data = df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig = px.bar(data, x=cat, y="count", template=tpl)
            insight = f"Top segment: {data.iloc[0][cat]}."
        else:
            agg = df.groupby(cat)["SpendPerServing"].mean().reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", template=tpl)
            insight = f"{agg.loc[agg.SpendPerServing.idxmax(),cat]} spends highest."
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Insight:** {insight}")
    pdf_insights["Visualization"] = insight
    pdf_figs["Visualization"] = fig

# Classification
elif page == "Classification":
    st.header("🤖 Classification")
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xt); Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    if algo == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    elif algo == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algo == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()
    model.fit(Xt_s, yt); preds = model.predict(Xe_s); probs = model.predict_proba(Xe_s)[:,1]
    mets = {
        "Accuracy": accuracy_score(ye, preds),
        "Precision": precision_score(ye, preds),
        "Recall": recall_score(ye, preds),
        "F1-score": f1_score(ye, preds)
    }
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))
    cm = confusion_matrix(ye, preds)
    fig_cm = go.Figure(go.Heatmap(z=cm, x=["Pred No","Pred Yes"], y=["Act No","Act Yes"],
                                  text=cm, texttemplate="%{text}", colorscale="Blues"))
    fig_cm.update_layout(template=tpl, title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)
    fpr, tpr, _ = roc_curve(ye, probs)
    fig_roc = px.area(x=fpr, y=tpr, title="ROC Curve", labels={'x':'FPR','y':'TPR'}, template=tpl)
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)
    insight = f"{algo} accuracy: {mets['Accuracy']:.2f}"
    pdf_insights["Classification"] = insight
    pdf_figs["Confusion Matrix"] = fig_cm
    pdf_figs["ROC Curve"] = fig_roc

# Clustering
elif page == "Clustering":
    st.header("🤝 Clustering")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    inertias = [KMeans(n_clusters=k, random_state=42).fit(df[feats]).inertia_ for k in range(2,11)]
    fig_elbow = px.line(x=list(range(2,11)), y=inertias, markers=True, labels={'x':'K','y':'Inertia'}, template=tpl)
    st.plotly_chart(fig_elbow, use_container_width=True)
    k_val = st.slider("Clusters (K)", 2, 8, 4)
    km = KMeans(n_clusters=k_val, random_state=42).fit(df[feats]); df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig_centers = px.bar(centers, x=centers.index, y=feats, barmode='group', template=tpl)
    st.plotly_chart(fig_centers, use_container_width=True)
    prof = df.groupby("Cluster")[feats].mean().round(1)
    for c, row in prof.iterrows():
        st.markdown(f"**Cluster {c}:** " + ", ".join(f"{f}={v}" for f, v in row.items()))
    insight = f"Elbow suggests K={k_val} clusters."
    pdf_insights["Clustering"] = insight
    pdf_figs["Elbow Method"] = fig_elbow
    pdf_figs["Cluster Centers"] = fig_centers

# Association
elif page == "Association":
    st.header("🔗 Association Rules")
    cols = [c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    s = st.slider("Min Support", 0.01, 0.2, 0.05)
    cth = st.slider("Min Confidence", 0.1, 0.7, 0.3)
    freq = apriori(df[cols], min_support=s, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=cth)
    rules["rule"] = rules.apply(lambda r: ", ".join(r.antecedents) + " → " + ", ".join(r.consequents), axis=1)
    top = rules.sort_values("lift", ascending=False).head(10)
    fig = px.bar(top, x="lift", y="rule", orientation='h', template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    insight = f"Top rule: {top.iloc[0]['rule']}."
    pdf_insights["Association"] = insight
    pdf_figs["Top Association Rules"] = fig

# Anomaly
elif page == "Anomaly":
    st.header("🚨 Anomaly Detection")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso = IsolationForest(contamination=0.05, random_state=42).fit(df[feats]); df["Anomaly"] = iso.predict(df[feats])
    fig = px.scatter(df, x="MonthlyDisposableIncome", y="SpendPerServing",
                     color=df["Anomaly"].map({1:"Normal",-1:"Anomaly"}), template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    insight = f"Outliers detected: {(df.Anomaly==-1).sum()}."
    pdf_insights["Anomaly"] = insight
    pdf_figs["Anomaly Scatter"] = fig

# Regression
elif page == "Regression":
    st.header("📈 Regression Comparison")
    Xr, yr = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]], df["SpendPerServing"]
    Xt, Xe, yt, ye = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    models = {"Linear":LinearRegression(), "Lasso":Lasso(), "Ridge":Ridge(), "Tree":DecisionTreeRegressor()}
    res = []
    for name, m in models.items():
        m.fit(Xt, yt); pr=m.predict(Xe)
        res.append({"Model":name, "R2":r2_score(ye, pr), "RMSE":np.sqrt(mean_squared_error(ye, pr))})
    rdf = pd.DataFrame(res)
    st.table(rdf)
    fig = px.bar(rdf, x="Model", y=["R2","RMSE"], barmode='group', template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    insight = "Compared regression models."
    pdf_insights["Regression"] = insight
    pdf_figs["Regression Comparison"] = fig

# Forecasting
elif page == "Forecasting":
    st.header("⏱️ Forecasting")
    ts = df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    if len(ts)>30:
        m = Prophet().fit(ts)
        fut = m.make_future_dataframe(periods=30); fc = m.predict(fut)
        fig = plot_plotly(m, fc)
        st.plotly_chart(fig, use_container_width=True)
        insight = f"Next week avg spend: AED {fc.yhat[-7:].mean():.2f}"
        pdf_insights["Forecasting"] = insight
        pdf_figs["Forecast"] = fig
    else:
        st.warning("Not enough data to forecast.")

# Cohort
elif page == "Cohort":
    st.header("👥 Cohort Analysis")
    mode = st.radio("Cohort by:", ["Time","Cluster"])
    if mode=="Time":
        df["Month"] = df.SurveyDate.dt.to_period("M").astype(str)
        cr = df.groupby("Month")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.line(cr, x="Month", y="Rate", template=tpl)
        st.plotly_chart(fig, use_container_width=True)
        insight = "Subscription rate over time."
        pdf_insights["Cohort Time"] = insight
        pdf_figs["Cohort Time"] = fig
    else:
        feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
        k = st.slider("Clusters for Cohort",2,8,4)
        cl = KMeans(n_clusters=k, random_state=42).fit_predict(df[feats]); df["CohortCluster"]=cl
        cr = df.groupby("CohortCluster")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig = px.bar(cr, x="CohortCluster", y="Rate", template=tpl)
        st.plotly_chart(fig, use_container_width=True)
        insight = "Subscription rate by cluster."
        pdf_insights["Cohort Cluster"] = insight
        pdf_figs["Cohort Cluster"] = fig

# Geography
elif page == "Geography":
    st.header("🗺️ Geography")
    stats = df.groupby("City")["SpendPerServing"].mean().reset_index().sort_values("SpendPerServing", ascending=False)
    fig = px.choropleth(locations=stats['City'], locationmode='country names',
                        color=stats['SpendPerServing'], template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    insight = "Spend by city."
    pdf_insights["Geography"] = insight
    pdf_figs["Geography Map"] = fig

# Sentiment
elif page == "Sentiment":
    st.header("💬 Sentiment Analysis")
    text = " ".join(df.Feedback.astype(str))
    wc = WordCloud(background_color="#333" if is_dark else "#fff").generate(text)
    fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(wc); ax.axis("off")
    st.pyplot(fig)
    sent = df.Feedback.astype(str).apply(lambda x:TextBlob(x).sentiment.polarity)
    labels = pd.cut(sent, [-1,-0.1,0.1,1], labels=["Negative","Neutral","Positive"])
    cnt = labels.value_counts()
    fig2 = px.pie(names=cnt.index, values=cnt.values, template=tpl)
    st.plotly_chart(fig2, use_container_width=True)
    insight = "Overall sentiment distribution."
    pdf_insights["Sentiment"] = insight
    pdf_figs["Word Cloud"] = fig
    pdf_figs["Sentiment Pie"] = fig2

# LTV & Churn
elif page == "LTV & Churn":
    st.header("💰 LTV & Churn")
    df["FreqNum"] = df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"] = df.SpendPerServing * df.FreqNum * 12
    churn = (df.SubscribePlan=="No").astype(int)
    Xc = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt, Xe, yt, ye = train_test_split(Xc, churn, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42).fit(Xt, yt)
    pr = clf.predict(Xe)
    mets = {"Accuracy":accuracy_score(ye,pr),"Precision":precision_score(ye,pr),"Recall":recall_score(ye,pr)}
    st.table(pd.DataFrame.from_dict(mets,orient='index',columns=['Value']))
    fig = px.histogram(df, x="LTV", template=tpl)
    st.plotly_chart(fig, use_container_width=True)
    insight = "LTV distribution and churn model metrics."
    pdf_insights["LTV & Churn"] = insight
    pdf_figs["LTV Histogram"] = fig

# Price Elasticity
elif page == "Price Elasticity":
    st.header("💵 Price Elasticity Simulator")
    price = st.slider("Price per Serving (AED)", 5,30,12)
    buyers = df[df.SpendPerServing>=price].shape[0]
    rev = buyers*price
    st.metric("Projected Buyers", buyers)
    st.metric("Projected Revenue (AED)", rev)
    insight = f"At price {price}, {buyers} buyers, revenue {rev}."
    pdf_insights["Price Elasticity"] = insight

# Chat
elif page == "Chat":
    st.header("💬 ChatGPT Q&A")
    try:
        import openai
        if "msgs" not in st.session_state:
            st.session_state.msgs=[{"role":"system","content":"You are an assistant."}]
        query=st.chat_input("Ask about data/code")
        if query:
            st.session_state.msgs.append({"role":"user","content":query})
            openai.api_key=st.secrets.get("OPENAI_API_KEY","")
            resp=openai.chat.completions.create(model="gpt-3.5-turbo",messages=st.session_state.msgs[-6:])
            msg=resp.choices[0].message
            st.session_state.msgs.append({"role":msg.role,"content":msg.content})
        for m in st.session_state.msgs:
            st.chat_message(m["role"]).write(m["content"])
    except ImportError:
        st.warning("Install openai to enable chat.")

# Glossary
elif page == "Glossary":
    st.header("📚 Glossary")
    glossary={
        "SpendPerServing":"AED per serving","HealthConsciousness":"Health importance 1–10",
        "TryNewBrand":"Will try new brand","SubscribePlan":"Subscription status",
        "MonthlyDisposableIncome":"Disposable income/month","Age":"Age",
        "ExerciseFrequency":"Frequency of exercise","ConsumptionFrequency":"Consumption frequency",
        "Feedback":"User feedback text"
    }
    for term, desc in glossary.items():
        st.markdown(f"**{term}**: {desc}")

# Per-page PDF download
if st.button("📄 Download This Page (PDF)"):
    pdf_bytes = generate_pdf(pdf_insights, pdf_figs)
    st.download_button("Save PDF", pdf_bytes, f"{page}_analysis.pdf", "application/pdf")
