
import streamlit as st
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/health_drink_survey_1000_augmented.csv", parse_dates=["SurveyDate"])
df = load_data()

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Ultimate Integrated")

# Sidebar filters
st.sidebar.header("Filters & Data")
# Demographics
age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
df = df[df.Age.between(age_range[0], age_range[1])]
genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
df = df[df.Gender.isin(genders)]
occupations = st.sidebar.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
df = df[df.Occupation.isin(occupations)]
# Behavior
ex_freq = st.sidebar.multiselect("Exercise Frequency", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
df = df[df.ExerciseFrequency.isin(ex_freq)]
cons_freq = st.sidebar.multiselect("Consumption Frequency", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
df = df[df.ConsumptionFrequency.isin(cons_freq)]
# Subscription
subs = st.sidebar.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
df = df[df.SubscribePlan.isin(subs)]
# City & Date
cities = st.sidebar.multiselect("City", df.City.unique(), df.City.unique())
df = df[df.City.isin(cities)]
date_range = st.sidebar.date_input("Survey Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
df = df[df.SurveyDate.between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

# Tabs
tabs = st.tabs(["Visualization","Classification","Clustering","Association","Anomaly","Regression","Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity"])

# 1. Visualization
with tabs[0]:
    st.header("1️⃣ Visualization")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if chart=="Scatter":
        x = st.selectbox("X-axis", num_cols, 0)
        y = st.selectbox("Y-axis", num_cols, 1)
        fig = px.scatter(df, x=x, y=y, color=cat_cols[0] if cat_cols else None, title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Correlation between {x} and {y}: {df[x].corr(df[y]):.2f}.")
    elif chart=="Histogram":
        col = st.selectbox("Feature", num_cols, 0)
        bins = st.slider("Bins", 10, 50, 30)
        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Mode of {col}: {df[col].mode()[0]}.")
    elif chart=="Box":
        cat = st.selectbox("Category", cat_cols, 0)
        val = st.selectbox("Value", num_cols, 0)
        fig = px.box(df, x=cat, y=val, title=f"{val} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        med = df.groupby(cat)[val].median().to_dict()
        st.markdown(f"**Insight:** Median {val} by {cat}: {med}.")
    else:
        cat = st.selectbox("Category", cat_cols, 0)
        metric = st.selectbox("Metric", ["Count","Mean Spend","Sum Spend"], 0)
        if metric=="Count":
            data = df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig = px.bar(data, x=cat, y="count", title=f"Count by {cat}")
        else:
            agg = df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", title=f"{metric} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** {metric} across {cat}.")

# 2. Classification
with tabs[1]:
    st.header("2️⃣ Classification")
    X = df.select_dtypes(include=[np.number]).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s,Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = KNeighborsClassifier() if algo=="KNN" else DecisionTreeClassifier() if algo=="Decision Tree" else RandomForestClassifier() if algo=="Random Forest" else GradientBoostingClassifier()
    model.fit(Xtr_s,ytr)
    yp = model.predict(Xte_s); yp_prob = model.predict_proba(Xte_s)[:,1]
    st.json({"Accuracy":accuracy_score(yte,yp),"Precision":precision_score(yte,yp),"Recall":recall_score(yte,yp),"F1":f1_score(yte,yp)})
    cm = confusion_matrix(yte,yp); fig=go.Figure(data=go.Heatmap(z=cm,x=["No","Yes"],y=["No","Yes"],colorscale="Viridis"))
    st.plotly_chart(fig, use_container_width=True)
    fpr,tpr,_=roc_curve(yte,yp_prob); roc_fig=go.Figure(data=go.Scatter(x=fpr,y=tpr,mode='lines')); roc_fig.update_layout(title="ROC Curve")
    st.plotly_chart(roc_fig, use_container_width=True)
    st.markdown("**Insight:** Model performance metrics guide targeting strategies.")

# 3. Clustering
with tabs[2]:
    st.header("3️⃣ Clustering")
    feats=['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    k=st.slider("Clusters (k)",2,8,4)
    km=KMeans(n_clusters=k,random_state=42).fit(df[feats])
    df['Cluster']=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=feats).round(2)
    fig=px.bar(centers, x=centers.index, y=feats, barmode='group',title="Cluster Centers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Identify distinct personas for tailored marketing.")

# 4. Association Rules
with tabs[3]:
    st.header("4️⃣ Association Rules")
    cols=[c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    sup=st.slider("Min Support",0.01,0.2,0.05); conf=st.slider("Min Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=sup,use_colnames=True); rules=association_rules(freq,metric="confidence",min_threshold=conf)
    rules['rule']=rules['antecedents'].apply(lambda x:", ".join(x))+" → "+rules['consequents'].apply(lambda x:", ".join(x))
    top=rules.sort_values('lift',ascending=False).head(10)
    fig=px.bar(top,x='lift',y='rule',orientation='h',color='confidence',color_continuous_scale='Viridis',title="Top Rules by Lift")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insight:** Bundle offerings based on common purchase contexts/flavours.")

# 5. Anomaly
with tabs[4]:
    st.header("5️⃣ Anomaly Detection")
    feats=['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    iso=IsolationForest(contamination=0.05,random_state=42).fit(df[feats]); df['Anom']=iso.predict(df[feats])
    fig=px.scatter(df,x='MonthlyDisposableIncome',y='SpendPerServing',color=df['Anom'].map({1:'Normal',-1:'Anomaly'}),title="Anomalies")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insight:** High spending anomalies indicate premium segments.")

# 6. Regression
with tabs[5]:
    st.header("6️⃣ Regression")
    Xr=df[['MonthlyDisposableIncome','HealthConsciousness','Age']]; yr=df['SpendPerServing']
    Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    reg=RandomForestRegressor(random_state=42).fit(Xtr,ytr); pr=reg.predict(Xte)
    mse=mean_squared_error(yte,pr); rmse=np.sqrt(mse)
    st.json({"R2":r2_score(yte,pr),"RMSE":rmse})
    fig=px.scatter(x=yte,y=pr,labels={'x':'Actual','y':'Predicted'},title="Actual vs Predicted")
    st.plotly_chart(fig,use_container_width=True)

# 7. Forecasting
with tabs[6]:
    st.header("7️⃣ Time Series Forecasting")
    date_col = st.selectbox("Date column",["SurveyDate"])
    value_col = st.selectbox("Value column",["SpendPerServing"])
    ts_df = df[[date_col, value_col]].rename(columns={date_col:'ds', value_col:'y'}).dropna()
    m=Prophet(); m.fit(ts_df)
    future=m.make_future_dataframe(periods=30); forecast=m.predict(future)
    fig=plot_plotly(m,forecast); st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insight:** Projected Spend trend next month.")

# 8. Cohort Analysis
with tabs[7]:
    st.header("8️⃣ Cohort Analysis")
    df['CohortMonth']=df.SurveyDate.dt.to_period('M').astype(str)
    cohort=df.groupby('CohortMonth')['SubscribePlan'].apply(lambda x:(x=='Yes').mean()).reset_index(name='Rate')
    fig=px.line(cohort,x='CohortMonth',y='Rate',title="Monthly Subscription Rate by Cohort")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insight:** Retention of subscription over time.")

# 9. Geographic View
with tabs[8]:
    st.header("9️⃣ Geographic Spend")
    city_avg=df.groupby('City')['SpendPerServing'].mean().reset_index()
    fig=px.bar(city_avg,x='City',y='SpendPerServing',title="Avg Spend per City")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insight:** Tailor campaigns to high-spend emirates.")

# 10. Sentiment
with tabs[9]:
    st.header("🔟 Sentiment Analysis")
    text=" ".join(df['Feedback'].astype(str))
    wc=WordCloud(width=800,height=400).generate(text)
    fig,ax=plt.subplots(figsize=(10,5)); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    st.markdown("**Insight:** Key consumer concerns from feedback.")

# 11. LTV & Churn
with tabs[10]:
    st.header("1️⃣1️⃣ LTV & Churn Prediction")
    df['FreqNum']=df.ConsumptionFrequency.map({'Never':0,'Rarely':1,'1-2':2,'3-4':4,'5+':5})
    df['LTV']=df.SpendPerServing*df.FreqNum*12
    churn_y = (df.SubscribePlan=='No').astype(int)
    Xc = df[['MonthlyDisposableIncome','HealthConsciousness','Age']]
    Xt,Xv,yt,yv = train_test_split(Xc,churn_y,test_size=0.2,random_state=42)
    clf=RandomForestClassifier(random_state=42).fit(Xt,yt)
    pred = clf.predict(Xv)
    st.json({"Accuracy":accuracy_score(yv,pred),"Precision":precision_score(yv,pred),"Recall":recall_score(yv,pred)})
    st.markdown("**Insight:** LTV distribution and churn model metrics inform retention strategies.")
    fig = px.histogram(df, x='LTV', nbins=30, title="LTV Distribution"); st.plotly_chart(fig,use_container_width=True)

# 12. Price Elasticity
with tabs[11]:
    st.header("1️⃣2️⃣ Price Elasticity Simulator")
    price = st.slider("Price per Serving (AED)", 5, 30, 12)
    buyers = df[df.SpendPerServing>=price].shape[0]
    revenue = buyers*price
    st.metric("Estimated Buyers",buyers); st.metric("Projected Revenue (AED)",revenue)
    st.markdown("**Insight:** Find optimal price point balancing volume & margin.")
