
import streamlit as st
from streamlit_option_menu import option_menu
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path

# ---------------- Data Loading ----------------

# ---------------- Data Loading ----------------
def load_data(path_or_buf):
    df = pd.read_csv(path_or_buf, parse_dates=["SurveyDate"])
    return df

# Sidebar: choose data source or upload
local_path = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
if local_path.exists():
    data_source = str(local_path)
else:
    st.sidebar.warning("❗️Augmented data not found in 'data/'. Please upload CSV.")
    uploaded_file = st.sidebar.file_uploader("Upload augmented CSV", type="csv", key="data_upload")
    if uploaded_file:
        data_source = uploaded_file
    else:
        st.sidebar.stop()

df = load_data(data_source)


# ---------------- App Config ----------------
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Ultimate Integrated")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("🔎 Filters & Data")
# Demographics
if "Age" in df.columns:
    amin, amax = int(df.Age.min()), int(df.Age.max())
    age_range = st.sidebar.slider("Age range", amin, amax, (amin, amax))
    df = df[df.Age.between(*age_range)]

if "Gender" in df.columns:
    genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(genders)]

if "Occupation" in df.columns:
    occs = st.sidebar.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occs)]

# Behavior
if "ExerciseFrequency" in df.columns:
    exf = st.sidebar.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(exf)]

if "ConsumptionFrequency" in df.columns:
    cf = st.sidebar.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]

# Subscription
if "SubscribePlan" in df.columns:
    subs = st.sidebar.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(subs)]

# City & Date
if "City" in df.columns:
    cities = st.sidebar.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(cities)]

if "SurveyDate" in df.columns:
    drange = st.sidebar.date_input("Survey Date range",
                                   [df.SurveyDate.min().date(), df.SurveyDate.max().date()])
    df = df[df.SurveyDate.between(pd.to_datetime(drange[0]), pd.to_datetime(drange[1]))]

st.sidebar.download_button("⬇️ Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

# ---------------- Tabs ----------------

# ---- Top Navigation Menu ----
selected = option_menu(
    menu_title=None,
    options=["Visualization","Classification","Clustering","Association","Anomaly","Regression","Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity"],
    icons=["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up","clock-history","calendar","geo","chat-dots","cash-stack","cash-coin"],
    menu_icon="grid-3x3-gap",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding":"0!important","background-color":"#F0F3FB"},
        "icon": {"color":"#026EBD","font-size":"18px"},
        "nav-link": {"font-size":"16px","text-align":"center","margin":"0px 8px","--hover-color":"#E0E8F9"},
        "nav-link-selected": {"background-color":"#026EBD","color":"white"},
    }
)

if selected == 'Visualization':
    # 1️⃣ Visualization
elif selected == 'Visualization':
    st.header("Visualization")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if chart == "Scatter" and len(num_cols) >= 2:
        x = st.selectbox("X-axis", num_cols, 0)
        y = st.selectbox("Y-axis", num_cols, 1)
        color = st.selectbox("Color by (optional)", [None]+cat_cols, 0)
        fig = px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        corr = df[x].corr(df[y])
        st.markdown(f"**Insight:** Correlation between `{x}` & `{y}`: **{corr:.2f}**.")
    elif chart == "Histogram" and num_cols:
        col = st.selectbox("Feature", num_cols, 0)
        bins = st.slider("Bins", 10, 50, 30)
        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        mode = df[col].mode().iloc[0]
        st.markdown(f"**Insight:** Mode of `{col}` is **{mode}**.")
    elif chart == "Box" and num_cols and cat_cols:
        cat = st.selectbox("Category", cat_cols, 0)
        val = st.selectbox("Value", num_cols, 0)
        fig = px.box(df, x=cat, y=val, title=f"{val} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        med = df.groupby(cat)[val].median().to_dict()
        st.markdown(f"**Insight:** Median `{val}` by `{cat}`: {med}.")
    elif chart == "Bar" and cat_cols:
        cat = st.selectbox("Category", cat_cols, 0)
        metric = st.selectbox("Metric", ["Count","Mean Spend","Sum Spend"], 0)
        if metric == "Count":
            data = df[cat].value_counts().reset_index()
            data.columns = [cat,"count"]
            fig = px.bar(data, x=cat, y="count", title=f"Count by {cat}")
        else:
            agg = df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", title=f"{metric} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** `{metric}` across `{cat}`.")

# 2️⃣ Classification
elif selected == 'Classification':
    st.header("Classification")
    X = df.select_dtypes(include="number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (
        KNeighborsClassifier() if algo=="KNN" else
        DecisionTreeClassifier() if algo=="Decision Tree" else
        RandomForestClassifier() if algo=="Random Forest" else
        GradientBoostingClassifier()
    )
    model.fit(Xtr_s, ytr)
    yp = model.predict(Xte_s)
    yp_prob = model.predict_proba(Xte_s)[:,1]
    st.json({
        "Accuracy": accuracy_score(yte,yp),
        "Precision": precision_score(yte,yp),
        "Recall": recall_score(yte,yp),
        "F1": f1_score(yte,yp)
    })
    cm = confusion_matrix(yte,yp)
    fig = go.Figure(data=go.Heatmap(z=cm, x=["No","Yes"], y=["No","Yes"], colorscale="Viridis"))
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    fpr, tpr, _ = roc_curve(yte, yp_prob)
    roc_fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode="lines"))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(roc_fig, use_container_width=True)
    st.markdown("**Insight:** These metrics guide targeted marketing strategies.")

# 3️⃣ Clustering
elif selected == 'Clustering':
    st.header("Clustering")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    k = st.slider("Clusters (k)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(df[feats])
    df["Cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig = px.bar(centers, x=centers.index, y=feats, barmode="group", title="Cluster Centers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Identify distinct personas for personalized campaigns.")

# 4️⃣ Association Rules
elif selected == 'Association':
    st.header("Association Rules")
    cols = [c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    sup = st.slider("Min Support", 0.01, 0.2, 0.05)
    conf = st.slider("Min Confidence", 0.1, 0.7, 0.3)
    freq = apriori(df[cols], min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    rules["rule"] = rules["antecedents"].apply(lambda x: ", ".join(x)) + " → " + rules["consequents"].apply(lambda x: ", ".join(x))
    top = rules.sort_values("lift", ascending=False).head(10)
    fig = px.bar(top, x="lift", y="rule", orientation="h",  
                 color="confidence", color_continuous_scale="Viridis",  
                 title="Top Rules by Lift & Confidence")  
    st.plotly_chart(fig, use_container_width=True)  
    st.markdown("**Insight:** Bundle products based on strong associations.")  

# 5️⃣ Anomaly Detection
elif selected == 'Anomaly':
    st.header("Anomaly Detection")
    feats = ["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso = IsolationForest(contamination=0.05, random_state=42).fit(df[feats])
    df["Anomaly"] = iso.predict(df[feats])
    fig = px.scatter(df, x="MonthlyDisposableIncome", y="SpendPerServing",  
                     color=df["Anomaly"].map({1:"Normal",-1:"Anomaly"}),  
                     title="Anomaly Detection")  
    st.plotly_chart(fig, use_container_width=True)  
    st.markdown("**Insight:** High-value outliers may represent VIP customers.")  

# 6️⃣ Regression
elif selected == 'Regression':
    st.header("Regression")
    Xr = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    yr = df["SpendPerServing"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(random_state=42).fit(Xtr, ytr)
    pr = reg.predict(Xte)
    st.json({"R2": r2_score(yte,pr), "RMSE": np.sqrt(mean_squared_error(yte,pr))})
    fig = px.scatter(x=yte, y=pr, labels={"x":"Actual","y":"Predicted"}, title="Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)  
    st.markdown("**Insight:** Reliable spend forecasts aid pricing decisions.")

# 7️⃣ Time Series Forecasting
elif selected == 'Forecasting':
    st.header("Time Series Forecasting")
    ts = df[["SurveyDate","SpendPerServing"]].dropna().rename(columns={"SurveyDate":"ds","SpendPerServing":"y"})
    m = Prophet(); m.fit(ts)
    fut = m.make_future_dataframe(periods=30); fc = m.predict(fut)
    fig = plot_plotly(m, fc); st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Projected spend trend for next 30 days.")

# 8️⃣ Cohort Analysis
elif selected == 'Cohort':
    st.header("Cohort Analysis")
    df["CohortMonth"] = df.SurveyDate.dt.to_period("M").astype(str)
    cohort = df.groupby("CohortMonth")["SubscribePlan"].apply(lambda x: (x=="Yes").mean()).reset_index(name="Rate")
    fig = px.line(cohort, x="CohortMonth", y="Rate", title="Monthly Subscription Rate by Cohort")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Track subscription retention over time.")

# 9️⃣ Geographic View
elif selected == 'Geography':
    st.header("Geographic Spend")
    city_avg = df.groupby("City")["SpendPerServing"].mean().reset_index()
    fig = px.bar(city_avg, x="City", y="SpendPerServing", title="Avg Spend per City")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Focus marketing on high-spend emirates.")

# 🔟 Sentiment Analysis
elif selected == 'Sentiment':
    st.header("Sentiment Word Cloud")
    text = " ".join(df.Feedback.astype(str))
    wc = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig)
    st.markdown("**Insight:** Consumer barriers highlight areas for improvement.")

# 1️⃣1️⃣ LTV & Churn
elif selected == 'Classification':
    st.header("LTV & Churn Prediction")
    df["FreqNum"] = df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"] = df.SpendPerServing * df.FreqNum * 12
    churn = (df.SubscribePlan=="No").astype(int)
    Xc = df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt, Xv, yt, yv = train_test_split(Xc, churn, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42).fit(Xt, yt)
    pred = clf.predict(Xv)
    st.json({
        "Accuracy": accuracy_score(yv,pred),
        "Precision": precision_score(yv,pred),
        "Recall": recall_score(yv,pred)
    })
    fig = px.histogram(df, x="LTV", nbins=30, title="LTV Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** High LTV segments warrant VIP retention programs.")

# 1️⃣2️⃣ Price Elasticity
elif selected == 'Classification':
    st.header("Price Elasticity Simulator")
    price = st.slider("Price per Serving (AED)", 5, 30, 12)
    buyers = df[df.SpendPerServing >= price].shape[0]
    revenue = buyers * price
    st.metric("Estimated Buyers", buyers); st.metric("Projected Revenue (AED)", revenue)
    st.markdown("**Insight:** Adjust price to balance volume & margin.")
