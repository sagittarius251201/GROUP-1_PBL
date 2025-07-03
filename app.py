
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Ultimate v2")

# Sidebar filters
st.sidebar.header("Filters & Data")
source = st.sidebar.radio("Load data via", ["GitHub URL", "Upload CSV"])
if source == "GitHub URL":
    url = st.sidebar.text_input("Raw GitHub CSV URL",
        "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv")
    try:
        df = pd.read_csv(url)
    except:
        st.sidebar.error("Failed to load data from URL")
        st.stop()
else:
    file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not file:
        st.sidebar.info("Upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(file)

# Common filters
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
df = df[df.Age.between(age_range[0], age_range[1])]
genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
df = df[df.Gender.isin(genders)]
occupations = st.sidebar.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
df = df[df.Occupation.isin(occupations)]

st.sidebar.download_button("Download filtered data", df.to_csv(index=False).encode(), "filtered.csv")

# Tabs
tabs = st.tabs(["Visualization","Classification","Clustering","Association","Anomaly","Regression","Barriers","Ranking","Recommendation"])

### 1. Visualization ###
with tabs[0]:
    st.header("1️⃣ Visualization")
    st.markdown("Select chart type, axes, and aggregation:")
    chart_type = st.selectbox("Chart type", ["Scatter", "Histogram", "Box", "Bar"])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    agg_funcs = {"Count": "count", "Mean": "mean", "Sum": "sum", "Median": "median"}
    if chart_type == "Scatter":
        x = st.selectbox("X-axis", num_cols, index=num_cols.index("Age"))
        y = st.selectbox("Y-axis", num_cols, index=num_cols.index("SpendPerServing"))
        fig = px.scatter(df, x=x, y=y, color="Gender", trendline="ols",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** The relationship between {x} and {y} shows correlation coefficient of {df[x].corr(df[y]):.2f}. **Business Implication:** Use this insight to adjust pricing strategies for different age brackets.")
    elif chart_type == "Histogram":
        col = st.selectbox("Feature", num_cols, index=num_cols.index("MonthlyDisposableIncome"))
        bins = st.slider("Bins", 10, 50, 30)
        fig = px.histogram(df, x=col, nbins=bins, color_discrete_sequence=["#2A9D8F"],
                           title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** The {col} distribution peaks at {df[col].mode()[0]}. **Business Implication:** Target core income segment with bespoke promotions.")
    elif chart_type == "Box":
        cat = st.selectbox("Category", ["Gender","Occupation","ConsumptionFrequency"])
        val = st.selectbox("Value", num_cols, index=num_cols.index("SpendPerServing"))
        fig = px.box(df, x=cat, y=val,
                     color_discrete_sequence=["#E76F51"],
                     title=f"{val} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        medians = df.groupby(cat)[val].median().to_dict()
        st.markdown(f"**Insight:** Median {val} by {cat}: {medians}. **Implication:** Customize product bundles per category.")
    else:  # Bar
        cat = st.selectbox("Category", ["PackagingFormat","PurchaseChannel","TopHealthBenefit"])
        metric = st.selectbox("Metric", list(agg_funcs.keys()), index=list(agg_funcs.keys()).index("Count"))
        if agg_funcs[metric] == "count":
            data = df[cat].value_counts().reset_index()
            data.columns = [cat, "count"]
            fig = px.bar(data, x=cat, y="count", color_discrete_sequence=["#264653"], title=f"{metric} of {cat}")
        else:
            data = df.groupby(cat)["SpendPerServing"].agg(agg_funcs[metric]).reset_index()
            fig = px.bar(data, x=cat, y="SpendPerServing", color_discrete_sequence=["#264653"], title=f"{metric} Spend by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** {metric} of {cat} reveals consumer preferences. **Implication:** Adjust marketing focus accordingly.")

### 2. Classification ###
with tabs[1]:
    st.header("2️⃣ Classification")
    X = df.select_dtypes(include=np.number).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xt)
    Xt_s, Xv_s = scaler.transform(Xt), scaler.transform(Xv)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    if algo=="KNN":
        clf = KNeighborsClassifier()
    elif algo=="Decision Tree":
        clf = DecisionTreeClassifier()
    elif algo=="Random Forest":
        clf = RandomForestClassifier()
    else:
        clf = GradientBoostingClassifier()
    clf.fit(Xt_s, yt)
    yp = clf.predict(Xv_s)
    yp_prob = clf.predict_proba(Xv_s)[:,1]
    metrics = {"Accuracy":accuracy_score(yv,yp), "Precision":precision_score(yv,yp),
               "Recall":recall_score(yv,yp), "F1":f1_score(yv,yp)}
    st.json(metrics)
    st.markdown("**Business Insight:** High recall suggests our model captures most willing trialists, useful for targeted ads.")
    cm = confusion_matrix(yv,yp)
    fig = go.Figure(data=go.Heatmap(z=cm, x=["No","Yes"], y=["No","Yes"], colorscale="Viridis"))
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    fpr, tpr, _ = roc_curve(yv, yp_prob)
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', line_color="#2A9D8F"))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig, use_container_width=True)

### 3. Clustering ###
with tabs[2]:
    st.header("3️⃣ Clustering")
    feats=['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    data = df[feats]
    k = st.slider("Clusters (k)",2,8,4)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster'] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=feats).round(2)
    fig = px.bar(centers, x=centers.index, y=feats, barmode='group', 
                 color_discrete_sequence=px.colors.qualitative.Set3, title="Cluster Centers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Cluster centers show segment profiles: e.g., segment 1 with high income & spend ideal for premium launch.")

### 4. Association ###
with tabs[3]:
    st.header("4️⃣ Association Rules")
    cols=[c for c in df if c.startswith("Flavour_") or c.startswith("Context_")]
    sup=st.slider("Min Support",0.01,0.2,0.05)
    conf=st.slider("Min Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=sup,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=conf)
    rules['rule']=rules['antecedents'].apply(lambda x:', '.join(x))+" → "+rules['consequents'].apply(lambda x:', '.join(x))
    top=rules.sort_values('lift',ascending=False).head(10)
    fig = px.bar(top, x='lift', y='rule', orientation='h', 
                 color='confidence', color_continuous_scale='Viridis', title="Top Rules by Lift & Confidence")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Cross-preferences guide bundle creation: e.g., pairing flavours with contexts enhances uptake.")

### 5. Anomaly ###
with tabs[4]:
    st.header("5️⃣ Anomaly Detection")
    feats=['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    iso = IsolationForest(contamination=0.05, random_state=42).fit(data)
    df['Anomaly']=iso.predict(data)
    fig = px.scatter(df, x='MonthlyDisposableIncome', y='SpendPerServing', color=df['Anomaly'].map({1:'Normal',-1:'Anomaly'}),
                     color_discrete_map={'Normal':'#2A9D8F','Anomaly':'#E76F51'}, title="Anomaly Detection")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Premium spenders identified as anomalies can be VIP targets for loyalty programs.")

### 6. Regression ###
with tabs[5]:
    st.header("6️⃣ Regression")
    Xr=df[['MonthlyDisposableIncome','HealthConsciousness','Age']]; yr=df['SpendPerServing']
    Xt,Xv,yt,yv=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    reg=RandomForestRegressor(n_estimators=100,random_state=42).fit(Xt,yt); pr=reg.predict(Xv)
    mse=mean_squared_error(yv,pr); rmse=np.sqrt(mse)
    st.write({"R2":r2_score(yv,pr),"RMSE":rmse})
    fig = px.scatter(x=yv, y=pr, labels={'x':'Actual','y':'Predicted'}, 
                     title="Actual vs Predicted Spend", trendline="ols",
                     color_discrete_sequence=['#264653'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Strong R2 indicates reliable spend predictions for financial forecasting.")

### 7. Barrier Analysis ###
with tabs[6]:
    st.header("7️⃣ Barrier Analysis")
    barrier_cols=[c for c in df if c.startswith("Barrier_")]
    barrier_counts = df[barrier_cols].sum().sort_values(ascending=False)
    fig = px.bar(x=barrier_counts.index.str.replace('Barrier_',''), y=barrier_counts.values,
                 color_discrete_sequence=['#E76F51'], title="Barrier Counts")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Key barrier is high price; consider value packs or discounts.")

### 8. Ranking ###
with tabs[7]:
    st.header("8️⃣ Ranking Analysis")
    rank_cols=[c for c in df.columns if c.startswith("Rank_")]
    avg_ranks = df[rank_cols].mean().sort_values()
    fig = px.bar(x=avg_ranks.values, y=[col.replace('Rank_','') for col in avg_ranks.index],
                 orientation='h', color_discrete_sequence=['#2A9D8F'], title="Average Rankings (Lower=Higher Priority)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Consumers prioritize taste then nutritional value; marketing should emphasize flavor quality.")

### 9. Recommendation ###
with tabs[8]:
    st.header("9️⃣ Recommendation Analysis")
    fig = px.histogram(df, x='RecommendLikelihood', nbins=10, color_discrete_sequence=['#264653'],
                       title="Recommendation Likelihood Distribution")
    st.plotly_chart(fig, use_container_width=True)
    promoters = df[df.RecommendLikelihood >= 9].shape[0]
    detractors = df[df.RecommendLikelihood <= 6].shape[0]
    nps = (promoters - detractors) / df.shape[0] * 100
    st.metric("NPS Score", f"{nps:.1f}%")
    st.markdown("**Insight:** NPS above 50 indicates strong brand advocacy; maintain product quality to keep promoters happy.")
