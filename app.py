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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
from textblob import TextBlob
import networkx as nx
import openai
import time

# --- Streamlit & Theme Setup ---
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

# --- Sidebar Filters & Chips ---
filters = []
def safe_join(var, label):
    if isinstance(var, list) and var:
        filters.append(f"{label}: " + ", ".join(map(str,var)))

with st.sidebar.expander("Demographics", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    safe_join(st.multiselect("Gender", df.Gender.unique(), df.Gender.unique()), "Gender"); df = df[df.Gender.isin(st.session_state.get('Gender',df.Gender.unique()))]
    safe_join(st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique()), "Occ"); df = df[df.Occupation.isin(st.session_state.get('Occupation',df.Occupation.unique()))]

with st.sidebar.expander("Behavior"):
    safe_join(st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique()), "ExFreq"); df = df[df.ExerciseFrequency.isin(st.session_state.get('Exercise Freq',df.ExerciseFrequency.unique()))]
    safe_join(st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique()), "ConFreq"); df = df[df.ConsumptionFrequency.isin(st.session_state.get('Consumption Freq',df.ConsumptionFrequency.unique()))]

with st.sidebar.expander("Subscription & Location"):
    safe_join(st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique()), "SubPlan"); df = df[df.SubscribePlan.isin(st.session_state.get('Subscribe Plan',df.SubscribePlan.unique()))]
    safe_join(st.multiselect("City", df.City.unique(), df.City.unique()), "City"); df = df[df.City.isin(st.session_state.get('City',df.City.unique()))]

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]),pd.to_datetime(dr[1]))]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("Download Filtered", df.to_csv(index=False).encode(), "filtered.csv", "text/csv")
if filters:
    chips = " ".join(f"<span style='background:{hl};color:#FFF;padding:3px 8px;border-radius:4px;margin:2px'>{f}</span>" for f in filters)
    st.markdown("**Active Filters:** "+chips, unsafe_allow_html=True)

# --- Navigation ---
pages = ["Visualization","Classification","Clustering","Association","Anomaly","Regression",
         "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Chat","Glossary"]
icons = ["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
         "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","chat","book"]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal",
                   styles={"nav-link-selected":{"background-color":hl}})
st.markdown(f"**Home** → **{page}**")

# --- Page: Visualization (unchanged) ---
if page=="Visualization":
    # ... your existing code with template=tpl ...
    pass

# --- Page: Classification with ChatGPT & What-if ---
elif page=="Classification":
    X = df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y = LabelEncoder().fit_transform(df.TryNewBrand)
    Xt, Xe, yt, ye = train_test_split(X,y,test_size=0.2,random_state=42)
    sc = StandardScaler().fit(Xt)
    Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)
    from sklearn.neighbors import KNeighborsClassifier
    algo = st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (KNeighborsClassifier() if algo=="KNN" else
             DecisionTreeClassifier() if algo=="Decision Tree" else
             RandomForestClassifier() if algo=="Random Forest" else
             GradientBoostingClassifier())
    model.fit(Xt_s,yt); p=model.predict(Xe_s); prob=model.predict_proba(Xe_s)[:,1]
    mets = {"Accuracy":accuracy_score(ye,p),"Precision":precision_score(ye,p),
            "Recall":recall_score(ye,p),"F1":f1_score(ye,p)}
    st.table(pd.DataFrame.from_dict(mets,orient='index',columns=['Value']))

    # Feature Importance
    if hasattr(model,'feature_importances_'):
        imp = pd.Series(model.feature_importances_,index=X.columns).sort_values()
        fig=px.bar(imp,orientation='h',title="Feature Importance",
                   color=imp, color_continuous_scale='Viridis',
                   labels={'value':'Importance','index':'Feature'},template=tpl)
        st.plotly_chart(fig,use_container_width=True)

    # What-If
    with st.expander("What-If Analysis"):
        row = {c: st.slider(c,float(df[c].min()),float(df[c].max()),float(df[c].mean())) for c in X.columns}
        inp = pd.DataFrame([row])
        pred = model.predict(sc.transform(inp))[0]
        st.success("Will Try" if pred==1 else "Will Not Try")

    # Confusion Matrix
    cm = confusion_matrix(ye,p)
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred No","Pred Yes"], y=["Actual No","Actual Yes"],
        text=cm, texttemplate="%{text}", colorscale="Blues"),)
    fig.update_layout(title="Confusion Matrix",template=tpl)
    st.plotly_chart(fig,use_container_width=True)

    fpr,tpr,_ = roc_curve(ye,prob)
    fig2=go.Figure(go.Scatter(x=fpr,y=tpr,mode='lines',line_color=hl))
    fig2.update_layout(title="ROC Curve",template=tpl)
    st.plotly_chart(fig2,use_container_width=True)

# --- Page: Clustering with Elbow & Personas ---
elif page=="Clustering":
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    st.subheader("Elbow Method")
    inertias=[]; Ks=range(2,11)
    for k in Ks:
        inertias.append(KMeans(n_clusters=k,random_state=42).fit(df[feats]).inertia_)
    fig=px.line(x=list(Ks),y=inertias,markers=True,
                labels={'x':'K','y':'Inertia'},template=tpl)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("Select optimal K by the elbow point.")

    k = st.slider("Clusters (K)",2,8,4)
    km = KMeans(n_clusters=k,random_state=42).fit(df[feats])
    df["Cluster"]=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=feats).round(2)
    fig2=px.bar(centers,x=centers.index,y=feats,barmode='group',template=tpl)
    st.plotly_chart(fig2,use_container_width=True)

    # Personas
    prof = df.groupby("Cluster")[feats].mean().round(1)
    st.subheader("Cluster Profiles")
    for c,row in prof.iterrows():
        st.info(f"Cluster {c}: "+", ".join(f"{f}={v}" for f,v in row.items()))

    # Compare
    a,b = st.selectbox("C1",prof.index), st.selectbox("C2",prof.index, key="c2")
    diff=(prof.loc[a]-prof.loc[b]).to_frame("∆").T
    st.table(diff)

# --- Page: Association with Network ---
elif page=="Association":
    # ... your existing bar chart ...
    top = association_rules(apriori(df.filter(like="Flavour_"),min_support=0.05,use_colnames=True),
                             metric="confidence",min_threshold=0.3).sort_values("lift",ascending=False).head(8)
    G=nx.Graph()
    for _,r in top.iterrows():
        for x in r.antecedents:
            for y in r.consequents:
                G.add_edge(x,y)
    st.subheader("Rule Network")
    pos=nx.spring_layout(G)
    edge_x=[];edge_y=[]
    for e in G.edges():
        x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
        edge_x+= [x0,x1,None]; edge_y+=[y0,y1,None]
    edge_trace=go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(width=1),hoverinfo='none')
    node_x=[];node_y=[]
    for n in G.nodes():
        x,y=pos[n]; node_x.append(x); node_y.append(y)
    node_trace=go.Scatter(x=node_x,y=node_y,mode='markers+text',text=list(G.nodes()),
                          textposition='top center',marker=dict(size=20,color=hl))
    fig=go.Figure([edge_trace,node_trace],layout=go.Layout(template=tpl))
    st.plotly_chart(fig,use_container_width=True)

# --- Page: Cohort with Time vs Cluster toggle ---
elif page=="Cohort":
    mode = st.radio("Cohort Type", ["Time","Cluster"])
    if mode=="Time":
        df["Month"]=df.SurveyDate.dt.to_period("M").astype(str)
        cr=df.groupby("Month")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig=px.line(cr,x="Month",y="Rate",template=tpl); st.plotly_chart(fig,use_container_width=True)
    else:
        if "Cluster" not in df: st.error("Run Clustering first."); st.stop()
        cr=df.groupby("Cluster")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig=px.bar(cr,x="Cluster",y="Rate",template=tpl); st.plotly_chart(fig,use_container_width=True)

# --- Page: ChatGPT Interactive Q&A ---
elif page=="Chat":
    st.header("💬 Ask GPT about your Data or Code")
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{"role":"system","content":"You are a data analyst assistant."}]
    query = st.chat_input("What do you want to know?")
    if query:
        st.session_state.msgs.append({"role":"user","content":query})
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.msgs
            )
            assistant_msg = resp["choices"][0]["message"]
            st.session_state.msgs.append({
                "role": assistant_msg["role"],
                "content": assistant_msg["content"]
            })
        except openai.error.RateLimitError:
            st.error("⚠️ Rate limit exceeded. Please wait a moment and try again.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

    for m in st.session_state.msgs:
        st.chat_message(m["role"]).write(m["content"])



# --- Page: Glossary ---
elif page=="Glossary":
    st.header("📚 Data Glossary")
    gloss = {
        "SpendPerServing":"AED spent per serving.",
        "HealthConsciousness":"Self-rated health priority (1-10).",
        "TryNewBrand":"Willingness to try new brand (Yes/No).",
        "SubscribePlan":"Subscription status.",
        "MonthlyDisposableIncome":"Monthly leftover income.",
        "Age":"Respondent age.",
        "ExerciseFrequency":"How often they exercise.",
        "ConsumptionFrequency":"How often they consume.",
        "Feedback":"Free-text feedback."
    }
    for k,v in gloss.items():
        st.markdown(f"**{k}**: {v}")

# --- Feedback Button ---
if st.button("💡 Send Feedback"):
    fb = st.text_area("Your feedback:")
    if st.button("Submit Feedback"):
        with open("feedback.txt","a") as f: f.write(fb+"\n---\n")
        st.success("Thanks for your feedback!")
