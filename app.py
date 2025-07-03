
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
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
from streamlit_option_menu import option_menu
import time

# Session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

# Theme switcher
theme = st.sidebar.radio("Theme", ['Light','Dark'], index=0)
st.session_state.theme = theme
# Apply CSS
if theme == 'Dark':
    st.markdown("""<style>
    body { background-color: #1B1B2F; color: #FFFFFF; }
    .sidebar .sidebar-content { background-color: #30475E; }
    h1, h2, h3, h4 { color: #EF476F; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
    body { background-color: #FFFFFF; color: #000000; }
    .sidebar .sidebar-content { background-color: #F8F9FB; }
    h1, h2, h3, h4 { color: #EF476F; }
    </style>""", unsafe_allow_html=True)

# Logo upload
logo = st.sidebar.file_uploader("Upload Logo", type=["png","jpg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

# Data loading with refresh
def load_data():
    local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
    if local.exists():
        path = local
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type="csv", key="data")
        if uploaded:
            path = uploaded
        else:
            st.stop()
    return pd.read_csv(path, parse_dates=["SurveyDate"])

if st.sidebar.button("Refresh Data"):
    df = load_data()
    st.sidebar.write(f"Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.title("🥤 Health Drink Survey Dashboard — Ultimate V3")

# Active filters chips
filters = []
if 'Age' in df.columns:
    amin, amax = int(df.Age.min()), int(df.Age.max())
    age = st.sidebar.slider("Age range", amin, amax, (amin, amax))
    df = df[df.Age.between(*age)]
    filters.append(f"Age: {age[0]}–{age[1]}")
# ... replicate for other filters ...
if 'Gender' in df.columns:
    g = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(g)]
    filters.append("Gender: " + ", ".join(g))

# Display filter chips
if filters:
    st.markdown("**Active Filters:** " + "  ".join([f"<span style='background-color:#EF476F;color:#FFF;padding:4px 8px;border-radius:4px'>{f}</span>" for f in filters]), unsafe_allow_html=True)

# Breadcrumb
page = option_menu(None,
    ["Visualization","Classification","Clustering","Association","Anomaly","Regression","Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Changelog"],
    icons=["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up","clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","file-text"],
    orientation="horizontal", menu_icon="cast", default_index=0,
    styles={"container":{"background-color":"#1B1B2F"},"nav-link":{"color":"#FFFFFF"}, "nav-link-selected":{"background-color":"#EF476F","color":"#FFFFFF"}}
)
st.markdown(f"**Home** > **{page}**")

# Pages implementation...
if page=="Visualization":
    st.header("📊 Visualization")
    # KPI Cards
    col1,col2,col3=st.columns(3)
    col1.metric("Avg Spend", f"{df.SpendPerServing.mean():.2f}")
    col2.metric("Trial Rate", f"{(df.TryNewBrand=='Yes').mean()*100:.1f}%", delta="↘2%")
    col3.metric("Subscribers", f"{(df.SubscribePlan=='Yes').sum()}")
    # Chart
    chart=st.selectbox("Chart", ["Scatter","Histogram"])
    if chart=="Scatter":
        fig=px.scatter(df, x="MonthlyDisposableIncome", y="SpendPerServing", color="Gender", opacity=0.7,
                       color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig,use_container_width=True)
    else:
        fig=px.histogram(df, x="SpendPerServing", nbins=30, color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig,use_container_width=True)

elif page=="Classification":
    st.header("🤖 Classification")
    # metrics table
    # ...
elif page=="Regression":
    st.header("📈 Regression Comparison")
    X=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]; y=df["SpendPerServing"]
    Xt,Xte,yt,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    models={"Lin":LinearRegression(),"Lasso":Lasso(),"Ridge":Ridge(),"Tree":DecisionTreeRegressor()}
    res=[]
    for name,mdl in models.items():
        mdl.fit(Xt,yt); pred=mdl.predict(Xte)
        res.append({"Model":name,"R2":r2_score(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred))})
    res_df=pd.DataFrame(res)
    st.table(res_df)
    fig=px.bar(res_df, x="Model", y=["R2","RMSE"], barmode="group", color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig, use_container_width=True)

elif page=="Changelog":
    st.header("📄 Changelog")
    path=Path(__file__).parent/"CHANGELOG.md"
    if path.exists():
        text=path.read_text()
        st.markdown(text)
    else:
        st.info("No changelog available.")

# Feedback
if st.button("💡 Feedback"):
    feedback=st.text_area("Your feedback here")
    if st.button("Submit Feedback"):
        with open("feedback.txt","a") as f:
            f.write(feedback+"\n---\n")
        st.success("Thanks for your feedback!")

