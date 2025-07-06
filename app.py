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

# ─── Page config & CSS ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")
st.markdown("""
<style>
.card {
  background-color: var(--card-bg);
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  margin-bottom: 20px;
}
.tooltip { border-bottom: 1px dotted var(--tooltip-fg); cursor: help; }
.light { --card-bg: #fff; --tooltip-fg: #555; }
.dark  { --card-bg: #333; --tooltip-fg: #ddd; }
</style>
""", unsafe_allow_html=True)

# ─── Theme toggle ───────────────────────────────────────────────────────────────
if 'theme' not in st.session_state: st.session_state.theme = 'Light'
theme = st.sidebar.radio("Theme", ["Light","Dark"], index=0)
st.session_state.theme = theme
is_dark = (theme == "Dark")
tpl = "plotly_dark" if is_dark else "plotly_white"
st.markdown(f"<body class='{'dark' if is_dark else 'light'}'></body>", unsafe_allow_html=True)

# ─── Data loader ────────────────────────────────────────────────────────────────
def load_data():
    p = Path(__file__).parent/"data"/"health_drink_survey_1000_augmented.csv"
    if p.exists(): return pd.read_csv(p, parse_dates=["SurveyDate"])
    up = st.sidebar.file_uploader("Upload data CSV", type="csv")
    if up: return pd.read_csv(up, parse_dates=["SurveyDate"])
    st.sidebar.error("No data file found."); st.stop()

if st.sidebar.button("🔄 Refresh Data"):
    df = load_data()
    st.sidebar.success(f"Refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

# ─── Sidebar filters ────────────────────────────────────────────────────────────
filters = []
def add_filter(label, sel):
    if sel:
        if isinstance(sel, list): filters.append(f"{label}: {', '.join(map(str, sel))}")
        else: filters.append(f"{label}: {sel}")

with st.sidebar.expander("Demographics", True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin,amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
    df = df[df.Gender.isin(g)]; add_filter("Gender", g)
    occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
    df = df[df.Occupation.isin(occ)]; add_filter("Occupation", occ)

with st.sidebar.expander("Behavior"):
    ex = st.multiselect("Exercise Frequency", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
    df = df[df.ExerciseFrequency.isin(ex)]; add_filter("ExerciseFreq", ex)
    cf = st.multiselect("Consumption Frequency", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
    df = df[df.ConsumptionFrequency.isin(cf)]; add_filter("ConsumptionFreq", cf)

with st.sidebar.expander("Subscription & Location"):
    sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
    df = df[df.SubscribePlan.isin(sub)]; add_filter("SubscribePlan", sub)
    city = st.multiselect("City", df.City.unique(), df.City.unique())
    df = df[df.City.isin(city)]; add_filter("City", city)

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min().date(), df.SurveyDate.max().date()])
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        df = df[(df.SurveyDate >= start) & (df.SurveyDate <= end)]
        filters.append(f"Date: {dr[0]}–{dr[1]}")
# In your sidebar, after all the filters:
with st.sidebar.expander("🔎 Active Filters", expanded=False):
    filters = []
    if "age" in locals():
        filters.append(f"Age: {age[0]}–{age[1]}")
    if "gender" in locals() and gender:
        filters.append(f"Gender: {', '.join(gender)}")
    if "occ" in locals() and occ:
        filters.append(f"Occupation: {', '.join(occ)}")
    if "ex" in locals() and ex:
        filters.append(f"Exercise Freq: {', '.join(ex)}")
    if "cf" in locals() and cf:
        filters.append(f"Consumption Freq: {', '.join(cf)}")
    if "sub" in locals() and sub:
        filters.append(f"Subscribe Plan: {', '.join(sub)}")
    if "city" in locals() and city:
        filters.append(f"City: {', '.join(city)}")
    if "dr" in locals() and dr:
        filters.append(f"Date: {dr[0]}–{dr[1]}")
    if filters:
        st.write("\n".join(filters))
    else:
        st.write("No filters applied.")


st.sidebar.download_button("📥 Download Filtered CSV", df.to_csv(index=False).encode(), "filtered.csv")
if filters:
    chips = "  ".join(f"<span class='card tooltip' title='{f}'>{f}</span>" for f in filters)
    st.sidebar.markdown(f"**Active Filters:**<br>{chips}", unsafe_allow_html=True)

# ─── Navigation menu ────────────────────────────────────────────────────────────
pages = ["Visualization","Classification","Clustering","Association","Anomaly","Regression",
         "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Chat","Glossary"]
icons = ["bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
         "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","chat","book"]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal",
                   styles={"nav-link-selected":{"background-color":"#EF476F"}})
st.markdown(f"**Home** → **{page}**")

# ─── 1. Visualization ──────────────────────────────────────────────────────────
if page=="Visualization":
    st.header("📊 Visualization")
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='card'><h3>Avg Spend</h3><h2>AED {df.SpendPerServing.mean():.2f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><h3>Trial Rate</h3><h2>{(df.TryNewBrand=='Yes').mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><h3>Subscribers</h3><h2>{(df.SubscribePlan=='Yes').sum()}</h2></div>", unsafe_allow_html=True)

    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()

    if chart=="Scatter":
        x=st.selectbox("X-axis",nums,0); y=st.selectbox("Y-axis",nums,1)
        fig=px.scatter(df,x=x,y=y,color=cats[0] if cats else None,opacity=0.7,template=tpl,color_continuous_scale="Turbo")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"**Insight:** Correlation between **{x}** and **{y}**.")
    elif chart=="Histogram":
        col=st.selectbox("Feature",nums); fig=px.histogram(df,x=col,nbins=30,template=tpl,color_discrete_sequence=["#EF476F"])
        st.plotly_chart(fig,use_container_width=True); st.markdown(f"**Insight:** Mean **{col}** ≈ {df[col].mean():.2f}.")
    elif chart=="Box":
        cat=st.selectbox("Category",cats); val=st.selectbox("Value",nums)
        fig=px.box(df,x=cat,y=val,template=tpl,color_discrete_sequence=["#06D6A0"])
        st.plotly_chart(fig,use_container_width=True); st.markdown(f"**Insight:** Variation of **{val}** by **{cat}**.")
    else:
        cat=st.selectbox("Category",cats); metric=st.selectbox("Metric",["Count","Mean Spend"])
        if metric=="Count":
            data=df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig=px.bar(data,x=cat,y="count",template=tpl,color_discrete_sequence=["#FFD166"])
            st.plotly_chart(fig,use_container_width=True); st.markdown(f"**Insight:** Top: **{data.iloc[0][cat]}**.")
        else:
            agg=df.groupby(cat)["SpendPerServing"].mean().reset_index()
            fig=px.bar(agg,x=cat,y="SpendPerServing",template=tpl,color_discrete_sequence=["#EF476F"])
            st.plotly_chart(fig,use_container_width=True)
            top=agg.loc[agg.SpendPerServing.idxmax(),cat]; st.markdown(f"**Insight:** **{top}** spends most.")

# ─── 2. Classification ────────────────────────────────────────────────────────
elif page=="Classification":
    st.header("🤖 Classification")
    X=df.select_dtypes("number").drop(columns=["SpendPerServing"],errors="ignore")
    y=LabelEncoder().fit_transform(df.TryNewBrand)
    Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler().fit(Xt); Xt_s,Xe_s=sc.transform(Xt),sc.transform(Xe)
    algo=st.selectbox("Algorithm",["KNN","Decision Tree","Random Forest","GBRT"])
    model = (
        __import__('sklearn.neighbors',fromlist=['KNeighborsClassifier']).KNeighborsClassifier() if algo=="KNN" else
        DecisionTreeClassifier() if algo=="Decision Tree" else
        RandomForestClassifier() if algo=="Random Forest" else
        GradientBoostingClassifier()
    )
    model.fit(Xt_s,yt); preds=model.predict(Xe_s); probs=model.predict_proba(Xe_s)[:,1]
    mets={"Accuracy":accuracy_score(ye,preds),"Precision":precision_score(ye,preds),
          "Recall":recall_score(ye,preds),"F1-score":f1_score(ye,preds)}
    st.table(pd.DataFrame.from_dict(mets,orient='index',columns=['Value']))

    if hasattr(model,'feature_importances_'):
        imp=pd.Series(model.feature_importances_,index=X.columns).sort_values()
        fig=px.bar(imp,orientation='h',title="Feature Importance",template=tpl,
                   color=imp,color_continuous_scale="Viridis")
        st.plotly_chart(fig,use_container_width=True)
    cm=confusion_matrix(ye,preds)
    fig_cm=go.Figure(go.Heatmap(z=cm,x=["Pred No","Pred Yes"],y=["Act No","Act Yes"],
                                text=cm,texttemplate="%{text}",colorscale="Blues"))
    fig_cm.update_layout(title="Confusion Matrix",template=tpl)
    st.plotly_chart(fig_cm,use_container_width=True)
    fpr,tpr,_=roc_curve(ye,probs)
    fig_roc=px.area(x=fpr,y=tpr,title="ROC Curve",labels={'x':'FPR','y':'TPR'},template=tpl)
    fig_roc.add_shape(type='line',line=dict(dash='dash'),x0=0,x1=1,y0=0,y1=1)
    st.plotly_chart(fig_roc,use_container_width=True)
    with st.expander("🔮 What-If Analysis"):
        inp=pd.DataFrame([{c:st.slider(c,float(df[c].min()),float(df[c].max()),float(df[c].mean()))
                           for c in X.columns}])
        p=model.predict(sc.transform(inp))[0]
        st.success("✅ Will Try" if p==1 else "❌ Will Not Try")

# ─── 3. Clustering ────────────────────────────────────────────────────────────
elif page=="Clustering":
    st.header("🤝 Clustering")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    st.subheader("Elbow Method")
    inertias=[KMeans(n_clusters=k,random_state=42).fit(df[feats]).inertia_ for k in range(2,11)]
    fig_elbow=px.line(x=list(range(2,11)),y=inertias,markers=True,
                      labels={'x':'#Clusters','y':'Inertia'},template=tpl)
    st.plotly_chart(fig_elbow,use_container_width=True)
    k=st.slider("Clusters (K)",2,8,4)
    km=KMeans(n_clusters=k,random_state=42).fit(df[feats]); df["Cluster"]=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=feats).round(2)
    fig_centers=px.bar(centers,x=centers.index,y=feats,barmode='group',template=tpl)
    st.plotly_chart(fig_centers,use_container_width=True)
    st.subheader("Cluster Profiles")
    prof=df.groupby("Cluster")[feats].mean().round(1)
    for c,row in prof.iterrows():
        st.markdown(f"<div class='card'><b>Cluster {c}</b>: "+
                    ", ".join(f"{f}={v}" for f,v in row.items())+
                    "</div>",unsafe_allow_html=True)
    st.subheader("Compare Clusters")
    a=st.selectbox("Cluster A",prof.index); b=st.selectbox("Cluster B",prof.index,key="cmp")
    st.table((prof.loc[a]-prof.loc[b]).to_frame("Difference").T)
    st.download_button("📥 Download Cluster Labels",df.to_csv(index=False).encode(),"clusters.csv")

# ─── 4. Association Rule Mining ─────────────────────────────────────────────────
elif page=="Association":
    st.header("🔗 Association Rules")
    cols=[c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    s=st.slider("Min Support",0.01,0.2,0.05); cth=st.slider("Min Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=s,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=cth)
    rules["rule"]=rules.apply(lambda r:", ".join(r.antecedents)+" → "+", ".join(r.consequents),axis=1)
    top=rules.sort_values("lift",ascending=False).head(10)
    fig=px.bar(top,x="lift",y="rule",orientation='h',color="confidence",template=tpl)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"**Insight:** Strongest: **{top.iloc[0]['rule']}**")
    st.subheader("Network Graph")
    G=nx.Graph()
    for _,r in top.iterrows():
        for a in r.antecedents:
            for b in r.consequents:
                G.add_edge(a,b)
    pos=nx.spring_layout(G,seed=42)
    edge_x,edge_y=[],[]
    for e in G.edges():
        x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
        edge_x+=[x0,x1,None]; edge_y+=[y0,y1,None]
    edge_trace=go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(width=1))
    node_x,node_y=zip(*[pos[n] for n in G.nodes()])
    node_trace=go.Scatter(x=node_x,y=node_y,mode='markers+text',text=list(G.nodes()),
                          textposition='top center',marker=dict(size=20,color="#EF476F"))
    fig_net=go.Figure([edge_trace,node_trace],layout=go.Layout(template=tpl,title="Rule Network"))
    st.plotly_chart(fig_net,use_container_width=True)

# ─── 5. Anomaly Detection ───────────────────────────────────────────────────────
elif page=="Anomaly":
    st.header("🚨 Anomaly Detection")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]  
    iso=IsolationForest(contamination=0.05,random_state=42).fit(df[feats])
    df["Anomaly"]=iso.predict(df[feats])
    fig=px.scatter(df,x="MonthlyDisposableIncome",y="SpendPerServing",
                   color=df["Anomaly"].map({1:"Normal",-1:"Anomaly"}),
                   color_discrete_sequence=["#06D6A0","#EF476F"],template=tpl)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"**Outliers:** {(df.Anomaly==-1).sum()}")

# ─── 6. Regression ─────────────────────────────────────────────────────────────
elif page=="Regression":
    st.header("📈 Regression Comparison")
    Xr,yr=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]],df["SpendPerServing"]
    Xt,Xe,yt,ye=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    models={"Linear":LinearRegression(),"Lasso":Lasso(),"Ridge":Ridge(),"Tree":DecisionTreeRegressor()}
    res=[]
    for name,mdl in models.items():
        mdl.fit(Xt,yt); pr=mdl.predict(Xe)
        res.append({"Model":name,"R2":r2_score(ye,pr),"RMSE":np.sqrt(mean_squared_error(ye,pr))})
    rdf=pd.DataFrame(res)
    st.table(rdf)
    fig=px.bar(rdf,x="Model",y=["R2","RMSE"],barmode='group',template=tpl)
    st.plotly_chart(fig,use_container_width=True)

# ─── 7. Forecasting ────────────────────────────────────────────────────────────
elif page=="Forecasting":
    st.header("⏱️ Forecasting")
    ts=df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    if len(ts)>30:
        m=Prophet(); m.fit(ts)
        fut=m.make_future_dataframe(periods=30); fc=m.predict(fut)
        st.plotly_chart(plot_plotly(m,fc),use_container_width=True)
        st.markdown(f"**Next week avg spend:** AED {fc.yhat[-7:].mean():.2f}")
    else:
        st.warning("Not enough data to forecast.")

# ─── 8. Cohort ─────────────────────────────────────────────────────────────────
elif page=="Cohort":
    st.header("👥 Cohort Analysis")
    mode=st.radio("Cohort by:",["Time","Cluster"])
    if mode=="Time":
        df["Month"]=df.SurveyDate.dt.to_period("M").astype(str)
        cr=df.groupby("Month")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig=px.line(cr,x="Month",y="Rate",markers=True,template=tpl); st.plotly_chart(fig,use_container_width=True)
    else:
        feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
        k=st.slider("Clusters for Cohort",2,8,4,key="cohortk")
        cl=KMeans(n_clusters=k,random_state=42).fit_predict(df[feats])
        df["CohortCluster"]=cl
        cr=df.groupby("CohortCluster")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
        fig=px.bar(cr,x="CohortCluster",y="Rate",template=tpl); st.plotly_chart(fig,use_container_width=True)

# ─── 9. Geography ──────────────────────────────────────────────────────────────
elif page=="Geography":
    st.header("🗺️ Geography")
    stats=df.groupby("City")["SpendPerServing"].mean().reset_index().sort_values("SpendPerServing",ascending=False)
    fig=px.bar(stats,x="City",y="SpendPerServing",color="SpendPerServing",color_continuous_scale="Viridis",template=tpl)
    st.plotly_chart(fig,use_container_width=True)

# ─── 10. Sentiment ─────────────────────────────────────────────────────────────
elif page=="Sentiment":
    st.header("💬 Sentiment")
    text=" ".join(df.Feedback.astype(str))
    wc=WordCloud(width=800,height=400,background_color=("#333" if is_dark else "#fff")).generate(text)
    fig,ax=plt.subplots(figsize=(10,5)); ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig)
    sent=df.Feedback.astype(str).apply(lambda x:TextBlob(x).sentiment.polarity)
    labels=pd.cut(sent,[-1,-0.1,0.1,1],labels=["Negative","Neutral","Positive"])
    cnt=labels.value_counts()
    fig2=px.pie(names=cnt.index,values=cnt.values,template=tpl); st.plotly_chart(fig2,use_container_width=True)

# ─── 11. LTV & Churn ───────────────────────────────────────────────────────────
elif page=="LTV & Churn":
    st.header("💰 LTV & Churn")
    df["FreqNum"]=df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"]=df.SpendPerServing*df.FreqNum*12
    churn=(df.SubscribePlan=="No").astype(int)
    Xc=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt,Xe,yt,ye=train_test_split(Xc,churn,test_size=0.2,random_state=42)
    clf=RandomForestClassifier(random_state=42).fit(Xt,yt)
    pr=clf.predict(Xe)
    mets={"Accuracy":accuracy_score(ye,pr),"Precision":precision_score(ye,pr),"Recall":recall_score(ye,pr)}
    st.table(pd.DataFrame.from_dict(mets,orient='index',columns=['Value']))
    st.plotly_chart(px.histogram(df,x="LTV",nbins=30,template=tpl),use_container_width=True)
    prob=clf.predict_proba(Xe)[:,1]
    st.plotly_chart(px.histogram(prob,nbins=20,title="Churn Probability",template=tpl),use_container_width=True)

# ─── 12. Price Elasticity ──────────────────────────────────────────────────────
elif page=="Price Elasticity":
    st.header("💵 Price Elasticity")
    price=st.slider("Price per Serving",5,30,12)
    buyers=df[df.SpendPerServing>=price].shape[0]; rev=buyers*price
    st.metric("Projected Buyers",buyers); st.metric("Projected Revenue (AED)",rev)

# ─── 13. ChatGPT Q&A ───────────────────────────────────────────────────────────
elif page=="Chat":
    st.header("💬 DataGPT Chat")
    try:
        import openai
    except ImportError:
        st.warning("Add `openai` to requirements.txt to enable Chat.")
    else:
        if "msgs" not in st.session_state:
            st.session_state.msgs=[{"role":"system","content":"You are a data analyst assistant."}]
        query=st.chat_input("Ask about data or code...")
        if query:
            st.session_state.msgs.append({"role":"user","content":query})
            openai.api_key=st.secrets.get("OPENAI_API_KEY","")
            history=st.session_state.msgs[-6:]
            try:
                resp=openai.chat.completions.create(model="gpt-3.5-turbo",messages=history)
                msg=resp.choices[0].message
                st.session_state.msgs.append({"role":msg.role,"content":msg.content})
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
        for m in st.session_state.msgs:
            st.chat_message(m["role"]).write(m["content"])

# ─── 14. Glossary ─────────────────────────────────────────────────────────────
elif page=="Glossary":
    st.header("📚 Glossary")
    glossary={
        "SpendPerServing":"AED per serving","HealthConsciousness":"Health importance 1–10",
        "TryNewBrand":"Will try new brand","SubscribePlan":"Subscription status",
        "MonthlyDisposableIncome":"Disposable income/month","Age":"Age of respondent",
        "ExerciseFrequency":"Exercise frequency","ConsumptionFrequency":"Consumption frequency",
        "Feedback":"User feedback text"
    }
    for term,desc in glossary.items():
        st.markdown(f"**{term}**: {desc}")

# ─── Feedback ─────────────────────────────────────────────────────────────────
if st.button("💡 Send Feedback"):
    fb=st.text_area("Your feedback:")
    if st.button("Submit Feedback"):
        with open("feedback.txt","a") as f: f.write(fb+"\n---\n")
        st.success("Thanks for your feedback!")
