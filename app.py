
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
from prophet import Prophet
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Ultimate Monolithic")

# Sidebar: Data source & page selection
st.sidebar.header("1) Data & Navigation")
source = st.sidebar.radio("Load data via", ["GitHub URL", "Upload CSV"])
if source=="GitHub URL":
    url = st.sidebar.text_input("Raw GitHub CSV URL",
        "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv")
    try:
        df = pd.read_csv(url, parse_dates=True)
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not uploaded:
        st.sidebar.info("Please upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(uploaded, parse_dates=True)

page = st.sidebar.selectbox("Select Page", [
    "Visualization",
    "Time Series Forecasting",
    "LTV & Churn",
    "Price Elasticity",
    "Cohort Analysis",
    "Geographic Heatmap",
    "Sentiment Analysis",
    "Competitive Benchmark",
    "ROI Simulator"
])

# Common filters
if "Age" in df:
    age_min, age_max = int(df.Age.min()), int(df.Age.max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
    df = df[df.Age.between(age_range[0], age_range[1])]
if "Gender" in df:
    genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(genders)]
st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

# Page: Visualization
if page == "Visualization":
    st.header("Data Visualization")
    chart = st.selectbox("Chart Type", ["Scatter", "Histogram", "Box", "Bar"])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if chart=="Scatter":
        x = st.selectbox("X-axis", num_cols, index=0)
        y = st.selectbox("Y-axis", num_cols, index=1 if len(num_cols)>1 else 0)
        fig = px.scatter(df, x=x, y=y, color=df.columns[1] if df.columns[1] in df else None,
                         title=f"{y} vs {x}", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        corr = df[x].corr(df[y])
        st.markdown(f"**Insight:** Correlation between {x} and {y} is {corr:.2f}.")
    elif chart=="Histogram":
        col = st.selectbox("Feature", num_cols)
        bins = st.slider("Bins", 10, 50, 30)
        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        mode = df[col].mode().iloc[0]
        st.markdown(f"**Insight:** Most common {col} value is {mode}.")
    elif chart=="Box":
        cat = st.selectbox("Category", df.select_dtypes(exclude=[np.number]).columns.tolist())
        val = st.selectbox("Value", num_cols)
        fig = px.box(df, x=cat, y=val, title=f"{val} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        med = df.groupby(cat)[val].median().to_dict()
        st.markdown(f"**Insight:** Median {val} by {cat}: {med}.")
    else:
        cat = st.selectbox("Category", df.select_dtypes(exclude=[np.number]).columns.tolist())
        metric = st.selectbox("Metric", ["Count", "Mean Spend", "Sum Spend"])
        if metric=="Count":
            data = df[cat].value_counts().reset_index()
            data.columns=[cat,"count"]
            fig = px.bar(data, x=cat, y="count", title=f"Count by {cat}")
        else:
            agg = df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", title=f"{metric} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** {metric} across {cat}.")

# Page: Time Series Forecasting
elif page == "Time Series Forecasting":
    st.header("Time Series Forecasting")
    date_col = st.sidebar.text_input("Date column", df.columns[0])
    value_col = st.sidebar.text_input("Value column", "SpendPerServing")
    try:
        ts = df[[date_col, value_col]].dropna().rename(columns={date_col:"ds", value_col:"y"})
        m = Prophet()
        m.fit(ts)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Forecast trend for next 30 periods.")
    except Exception as e:
        st.error(f"Forecast error: {e}")

# Page: LTV & Churn
elif page == "LTV & Churn":
    st.header("LTV & Churn Prediction")
    st.markdown("Feature engineering and modeling for LTV & churn goes here.")

# Page: Price Elasticity
elif page == "Price Elasticity":
    st.header("Price Elasticity Simulator")
    price = st.sidebar.slider("Price per serving", 5, 30, 12)
    buyers = df[df['SpendPerServing']>=price].shape[0]
    st.metric("Estimated Buyers", buyers)
    revenue = buyers * price
    st.metric("Projected Revenue (AED)", revenue)
    st.markdown("**Insight:** Adjust price to optimize revenue.")

# Page: Cohort Analysis
elif page == "Cohort Analysis":
    st.header("Cohort Retention")
    if 'date' in df.columns:
        df['Cohort'] = df.groupby(df.columns[0])[ 'date' ].transform('min').dt.to_period('M')
        cohort = df.groupby(['Cohort', df['date'].dt.to_period('M')])['date'].count().reset_index()
        fig = px.line(cohort, x='date', y='date', color='Cohort', title="Retention Curves")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Cohort retention over time.")
    else:
        st.info("Date column not found.")

# Page: Geographic Heatmap
elif page == "Geographic Heatmap":
    st.header("Geographic Distribution")
    if 'city' in df.columns:
        fig = px.choropleth(df, locations='city', locationmode='USA-states', color='SpendPerServing',
                            title="Spend by City")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("City column not found.")

# Page: Sentiment Analysis
elif page == "Sentiment Analysis":
    st.header("Sentiment Word Cloud")
    if 'Feedback' in df.columns:
        text = " ".join(df['Feedback'].dropna().astype(str))
        wc = WordCloud(width=800, height=400).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("**Insight:** Common themes from feedback.")
    else:
        st.info("Feedback column not found.")

# Page: Competitive Benchmark
elif page == "Competitive Benchmark":
    st.header("Competitive Pricing")
    st.markdown("Upload competitor pricing CSV with columns Brand, Price")
    comp = st.file_uploader("Competitor CSV", type="csv")
    if comp:
        comp_df = pd.read_csv(comp)
        fig = px.bar(comp_df, x='Brand', y='Price', title="Competitor Prices")
        st.plotly_chart(fig, use_container_width=True)

# Page: ROI Simulator
elif page == "ROI Simulator":
    st.header("ROI Calculator")
    budget = st.number_input("Marketing Budget (AED)", 0)
    cpa = st.number_input("Cost per Acquisition (AED)", 10)
    if st.button("Calculate ROI"):
        acqu = budget / cpa if cpa>0 else 0
        revenue = acqu * df['SpendPerServing'].mean()
        roi = (revenue - budget) / budget * 100 if budget>0 else 0
        st.metric("Estimated ROI (%)", f"{roi:.1f}%")
        st.markdown("**Insight:** ROI based on average spend and acquisition cost.")
