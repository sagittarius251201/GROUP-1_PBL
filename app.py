
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

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Fixed")

# Sidebar: Data source & navigation
st.sidebar.header("Data & Navigation")
source = st.sidebar.radio("Load data via", ["GitHub URL", "Upload CSV"])
if source == "GitHub URL":
    url = st.sidebar.text_input("Raw GitHub CSV URL",
        "https://raw.githubusercontent.com/sagittarius251201/second-/refs/heads/main/health_drink_survey.csv")
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not uploaded:
        st.sidebar.info("Please upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(uploaded)

# Page selection
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
if 'Age' in df.columns:
    age_min, age_max = int(df.Age.min()), int(df.Age.max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
    df = df[df.Age.between(age_range[0], age_range[1])]
if 'Gender' in df.columns:
    genders = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(genders)]

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

# Visualization
if page == "Visualization":
    st.header("Visualization")
    chart = st.selectbox("Chart Type", ["Scatter", "Histogram", "Box", "Bar"])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if chart == "Scatter":
        x = st.selectbox("X-axis", num_cols, 0)
        y = st.selectbox("Y-axis", num_cols, min(1, len(num_cols)-1))
        fig = px.scatter(df, x=x, y=y, color=cat_cols[0] if cat_cols else None, title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Correlation ({x}, {y}) = {df[x].corr(df[y]):.2f}.")
    elif chart == "Histogram":
        col = st.selectbox("Feature", num_cols, 0)
        bins = st.slider("Bins", 10, 50, 30)
        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        mode = df[col].mode().iloc[0]
        st.markdown(f"**Insight:** Most common {col} = {mode}.")
    elif chart == "Box":
        cat = st.selectbox("Category", cat_cols, 0)
        val = st.selectbox("Value", num_cols, 0)
        fig = px.box(df, x=cat, y=val, title=f"{val} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Median {val} by {cat}: {df.groupby(cat)[val].median().to_dict()}.")
    else:
        cat = st.selectbox("Category", cat_cols, 0)
        metric = st.selectbox("Metric", ["Count", "Mean Spend", "Sum Spend"], 0)
        if metric == "Count":
            data = df[cat].value_counts().reset_index()
            data.columns = [cat, "count"]
            fig = px.bar(data, x=cat, y="count", title=f"Count by {cat}")
        else:
            agg = df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig = px.bar(agg, x=cat, y="SpendPerServing", title=f"{metric} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** {metric} across {cat}.")

# Forecasting
elif page == "Time Series Forecasting":
    st.header("Time Series Forecasting")
    all_cols = df.columns.tolist()
    date_col = st.selectbox("Date column", all_cols, 0)
    value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    value_col = st.selectbox("Value column", value_cols, 0)
    if date_col == value_col:
        st.error("Please select different date and value columns.")
    else:
        ds = pd.to_datetime(df[date_col], errors='coerce')
        df_ts = pd.DataFrame({'ds': ds, 'y': df[value_col]}).dropna()
        if df_ts.empty:
            st.error("No valid dates in selected column.")
        else:
            m = Prophet()
            m.fit(df_ts)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** Forecast next 30 periods.")

# LTV & Churn
elif page == "LTV & Churn":
    st.header("LTV & Churn")
    st.markdown("Feature engineering & models to calculate LTV and predict churn here.")

# Price Elasticity
elif page == "Price Elasticity":
    st.header("Price Elasticity Simulator")
    price = st.slider("Price", 5, 30, 12)
    buyers = df[df['SpendPerServing'] >= price].shape[0]
    st.metric("Estimated buyers", buyers)
    revenue = buyers * price
    st.metric("Projected revenue (AED)", revenue)
    st.markdown("**Insight:** Optimize price for revenue.")

# Cohort Analysis
elif page == "Cohort Analysis":
    st.header("Cohort Analysis")
    if 'date' in df.columns:
        df['Cohort'] = df.groupby(df.columns[0])['date'].transform('min').dt.to_period('M')
        cohort = df.groupby(['Cohort', df['date'].dt.to_period('M')])['date'].count().reset_index()
        fig = px.line(cohort, x='date', y='date', color='Cohort')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Retention curves by cohort.")
    else:
        st.info("No 'date' column found.")

# Geographic Heatmap
elif page == "Geographic Heatmap":
    st.header("Geographic Heatmap")
    if 'city' in df.columns:
        fig = px.choropleth(df, locations='city', color='SpendPerServing')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'city' column found.")

# Sentiment Analysis
elif page == "Sentiment Analysis":
    st.header("Sentiment Word Cloud")
    if 'Feedback' in df.columns:
        text = " ".join(df['Feedback'].dropna().astype(str))
        wc = WordCloud(width=800, height=400).generate(text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No 'Feedback' column found.")

# Competitive Benchmark
elif page == "Competitive Benchmark":
    st.header("Competitive Benchmark")
    comp = st.file_uploader("Competitor CSV", type="csv")
    if comp:
        comp_df = pd.read_csv(comp)
        fig = px.bar(comp_df, x='Brand', y='Price')
        st.plotly_chart(fig, use_container_width=True)

# ROI Simulator
elif page == "ROI Simulator":
    st.header("ROI Simulator")
    budget = st.number_input("Budget (AED)", 0)
    cpa = st.number_input("Cost per Acquisition", 10)
    if st.button("Calculate"):
        acqu = budget/cpa if cpa>0 else 0
        rev = acqu * df['SpendPerServing'].mean()
        roi = (rev - budget)/budget * 100 if budget>0 else 0
        st.metric("ROI (%)", f"{roi:.1f}%")
        st.markdown("**Insight:** ROI based on average spend.")

