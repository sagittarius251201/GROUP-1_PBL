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
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from textblob import TextBlob
import time
import networkx as nx

# --------- THEME LOGIC -------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")

theme = st.sidebar.radio("Theme", ['Light','Dark'], index=0)
st.session_state.theme = theme

if theme == 'Dark':
    bg_color = "#1B1B2F"
    fg_color = "#FFFFFF"
    sidebar_bg = "#30475E"
    highlight_color = "#EF476F"
    plotly_template = "plotly_dark"
else:
    bg_color = "#FFFFFF"
    fg_color = "#000000"
    sidebar_bg = "#F8F9FB"
    highlight_color = "#EF476F"
    plotly_template = "plotly_white"

st.markdown(f"""
    <style>
    body {{background-color:{bg_color}; color:{fg_color};}}
    .sidebar .sidebar-content {{background-color:{sidebar_bg};}}
    h1,h2,h3,h4 {{color:{highlight_color};}}
    </style>
""", unsafe_allow_html=True)

logo = st.sidebar.file_uploader("Upload Logo", type=["png","jpg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

def load_data():
    local = Path(__file__).parent / "data" / "health_drink_survey_1000_augmented.csv"
    if local.exists():
        return pd.read_csv(local, parse_dates=["SurveyDate"])
    upload = st.sidebar.file_uploader("Upload data CSV", type="csv", key="data")
    if upload:
        return pd.read_csv(upload, parse_dates=["SurveyDate"])
    st.sidebar.error("Data file not found.")
    st.stop()

if st.sidebar.button("Refresh Data"):
    df = load_data()
    st.sidebar.write(f"Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.experimental_rerun()

df = load_data()

filters = []
with st.sidebar.expander("Demographics", expanded=True):
    if "Age" in df:
        amin, amax = int(df.Age.min()), int(df.Age.max())
        age = st.slider("Age range", amin, amax, (amin, amax))
        df = df[df.Age.between(*age)]
        filters.append(f"Age: {age[0]}–{age[1]}")
    if "Gender" in df:
        g = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
        df = df[df.Gender.isin(g)]
        if isinstance(g, list) and len(g) > 0:
            filters.append("Gender: " + ", ".join([str(i) for i in g]))
    if "Occupation" in df:
        occ = st.multiselect("Occupation", df.Occupation.unique(), df.Occupation.unique())
        df = df[df.Occupation.isin(occ)]
        if isinstance(occ, list) and len(occ) > 0:
            filters.append("Occ: " + ", ".join([str(i) for i in occ]))

with st.sidebar.expander("Behavior"):
    if "ExerciseFrequency" in df:
        ex = st.multiselect("Exercise Freq", df.ExerciseFrequency.unique(), df.ExerciseFrequency.unique())
        df = df[df.ExerciseFrequency.isin(ex)]
        if isinstance(ex, list) and len(ex) > 0:
            filters.append("ExFreq: " + ", ".join([str(i) for i in ex]))
    if "ConsumptionFrequency" in df:
        cf = st.multiselect("Consumption Freq", df.ConsumptionFrequency.unique(), df.ConsumptionFrequency.unique())
        df = df[df.ConsumptionFrequency.isin(cf)]
        if isinstance(cf, list) and len(cf) > 0:
            filters.append("ConFreq: " + ", ".join([str(i) for i in cf]))

with st.sidebar.expander("Subscription & Location"):
    if "SubscribePlan" in df:
        sub = st.multiselect("Subscribe Plan", df.SubscribePlan.unique(), df.SubscribePlan.unique())
        df = df[df.SubscribePlan.isin(sub)]
        if isinstance(sub, list) and len(sub) > 0:
            filters.append("SubPlan: " + ", ".join([str(i) for i in sub]))
    if "City" in df:
        city = st.multiselect("City", df.City.unique(), df.City.unique())
        df = df[df.City.isin(city)]
        if isinstance(city, list) and len(city) > 0:
            filters.append("City: " + ", ".join([str(i) for i in city]))

with st.sidebar.expander("Survey Date"):
    if "SurveyDate" in df:
        dr = st.date_input("Date range", [df.SurveyDate.min(), df.SurveyDate.max()])
        df = df[df.SurveyDate.between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))]
        filters.append(f"Date: {dr[0]}–{dr[1]}")

st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False).encode(), "filtered.csv")

if filters:
    chips = " ".join([f"<span style='background:{highlight_color};color:#FFF;padding:4px 8px;border-radius:4px;margin:2px'>{f}</span>" for f in filters])
    st.markdown("**Active Filters:** " + chips, unsafe_allow_html=True)

pages = [
    "Visualization","Classification","Clustering","Association","Anomaly","Regression",
    "Forecasting","Cohort","Geography","Sentiment","LTV & Churn","Price Elasticity","Glossary"
]
icons = [
    "bar-chart","cpu","people","diagram-3","exclamation-triangle","graph-up",
    "clock-history","calendar","geo","chat-dots","cash-stack","cash-coin","book"
]
page = option_menu(None, pages, icons=icons, menu_icon="cast", orientation="horizontal", styles={"container":{"padding":"0px"},"nav-link":{"font-size":"14px"},"nav-link-selected":{"background-color":highlight_color}})
st.markdown(f"**Home** > **{page}**")

# 1. Visualization Tab -- unchanged (already covers basics)

if page=="Visualization":
    st.header("📊 Visualization")
    c1,c2,c3=st.columns(3)
    avg_spend = df.SpendPerServing.mean()
    c1.metric("Avg Spend", f"AED {avg_spend:.2f}")
    trial_rate = (df.TryNewBrand=='Yes').mean()*100
    c2.metric("Trial Rate", f"{trial_rate:.1f}%")
    nsubs = (df.SubscribePlan=='Yes').sum()
    c3.metric("Subscribers", f"{nsubs}")
    chart = st.selectbox("Chart Type", ["Scatter","Histogram","Box","Bar"])
    nums = df.select_dtypes("number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if chart=="Scatter":
        x=st.selectbox("X-axis", nums, 0); y=st.selectbox("Y-axis", nums, 1)
        fig=px.scatter(df, x=x, y=y, color=cats[0] if cats else None,
                       opacity=0.7, color_discrete_sequence=px.colors.sequential.Plasma, template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight:** Higher values of {y} are associated with higher {x} for {cats[0] if cats else 'all'} groups. Target users with high {x} for premium offerings.")
    elif chart=="Histogram":
        col=st.selectbox("Feature", nums)
        fig=px.histogram(df, x=col, nbins=30, color_discrete_sequence=px.colors.sequential.Viridis, template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
        most = df[col].mean()
        st.markdown(f"**Insight:** The average {col} is {most:.2f}. Use this value for targeting and pricing decisions.")
    elif chart=="Box":
        cat=st.selectbox("Category", cats)
        val=st.selectbox("Value", nums)
        fig=px.box(df, x=cat, y=val, color=cat, color_discrete_sequence=px.colors.sequential.Inferno, template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
        out = df[val][df[val]>df[val].quantile(0.75)].count()
        st.markdown(f"**Insight:** There are {out} above-average spenders in {cat}. Loyalty offers may be targeted here.")
    else:
        cat=st.selectbox("Category", cats)
        metric=st.selectbox("Metric", ["Count","Mean Spend","Sum Spend"])
        if metric=="Count":
            data=df[cat].value_counts().reset_index(); data.columns=[cat,"count"]
            fig=px.bar(data, x=cat, y="count", color_discrete_sequence=px.colors.sequential.Magma, template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            top = data.iloc[0][cat]
            st.markdown(f"**Insight:** The largest group is {top}. Tailor communication and distribution to this group.")
        else:
            agg=df.groupby(cat)["SpendPerServing"].agg("mean" if metric=="Mean Spend" else "sum").reset_index()
            fig=px.bar(agg, x=cat, y="SpendPerServing", color_discrete_sequence=px.colors.sequential.Magma, template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            hi = agg.sort_values("SpendPerServing", ascending=False).iloc[0][cat]
            st.markdown(f"**Insight:** {hi} segment drives the most spend. Prioritize in promotions.")

# 2. Classification Tab
elif page=="Classification":
    st.header("🤖 Classification")
    X=df.select_dtypes("number").drop(columns=["SpendPerServing"], errors="ignore")
    y=LabelEncoder().fit_transform(df.TryNewBrand)
    Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler().fit(Xt); Xt_s,Xe_s=sc.transform(Xt),sc.transform(Xe)
    from sklearn.neighbors import KNeighborsClassifier
    algo=st.selectbox("Algorithm", ["KNN","Decision Tree","Random Forest","GBRT"])
    model = (KNeighborsClassifier() if algo=="KNN" else
             DecisionTreeClassifier() if algo=="Decision Tree" else
             RandomForestClassifier() if algo=="Random Forest" else
             GradientBoostingClassifier())
    model.fit(Xt_s,yt); p=model.predict(Xe_s); prob=model.predict_proba(Xe_s)[:,1]
    mets={"Accuracy":accuracy_score(ye,p),"Precision":precision_score(ye,p),
          "Recall":recall_score(ye,p),"F1":f1_score(ye,p)}
    st.table(pd.DataFrame.from_dict(mets, orient='index', columns=['Value']))

    # --- Feature Importance
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig_feat = px.bar(feat_imp, orientation='h', title="Feature Importance",
                          color=feat_imp, color_continuous_scale='Viridis',
                          labels={'value':'Importance','index':'Feature'},
                          template=plotly_template)
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown(f"**Insight:** {feat_imp.idxmax()} is the most influential factor for predicting trial.")

    # --- What-If Analysis
    with st.expander("Try 'What If?' Prediction"):
        test_row = {}
        for col in X.columns:
            val = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            test_row[col] = val
        test_df = pd.DataFrame([test_row])
        test_df_scaled = sc.transform(test_df)
        pred = model.predict(test_df_scaled)[0]
        st.success(f"Prediction: {'Will Try' if pred==1 else 'Will Not Try'}")

    cm=confusion_matrix(ye,p)
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted No","Predicted Yes"], 
            y=["Actual No","Actual Yes"],
            colorscale="Plasma",
            showscale=True,
            text=cm, texttemplate="%{text}", 
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template=plotly_template,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    fpr,tpr,_=roc_curve(ye,prob)
    fig2=go.Figure(go.Scatter(x=fpr,y=tpr,mode='lines', line_color=highlight_color))
    fig2.update_layout(template=plotly_template, title="ROC Curve")
    st.plotly_chart(fig2, use_container_width=True)
    best_metric = max(mets, key=mets.get)
    st.markdown(f"**Insight:** {algo} model achieves the highest {best_metric} ({mets[best_metric]:.2f}). Use this for predicting who will try the product.")

# 3. Clustering Tab
elif page=="Clustering":
    st.header("🤝 Clustering")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    st.subheader("K-Means Elbow Method")
    inertias = []
    k_range = range(2, 11)
    for ki in k_range:
        km_tmp = KMeans(n_clusters=ki, random_state=42).fit(df[feats])
        inertias.append(km_tmp.inertia_)
    fig_elbow = px.line(
        x=list(k_range), y=inertias, 
        markers=True, template=plotly_template,
        labels={'x':'Number of Clusters (K)','y':'Inertia'},
        title="Elbow Method for Optimal K"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.markdown("**Insight:** The 'elbow' point suggests the optimal number of clusters for customer segmentation.")

    # Main KMeans
    k=st.slider("Clusters (K)",2,8,4)
    km=KMeans(n_clusters=k,random_state=42).fit(df[feats])
    df["Cluster"]=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=feats).round(2)
    fig=px.bar(centers, x=centers.index, y=feats, barmode="group",
               color_discrete_sequence=px.colors.qualitative.Plotly, template=plotly_template)
    st.plotly_chart(fig,use_container_width=True)
    sizes = df['Cluster'].value_counts().sort_values(ascending=False)
    top_c = sizes.index[0]
    st.markdown(f"**Insight:** Cluster {top_c} is the largest segment with {sizes.iloc[0]} consumers. Use this persona for new product launches.")

    # Cluster Profile Cards
    cluster_profiles = df.groupby('Cluster')[feats].mean().round(1)
    st.subheader("Cluster Profiles")
    for c in cluster_profiles.index:
        st.markdown(f"""
        <div style="border:2px solid {highlight_color};border-radius:10px;padding:8px;margin-bottom:8px;background:{'#f7f1ff' if theme=='Light' else '#222'}>
        <h4>Cluster {c}</h4>
        {" | ".join([f"<b>{k}:</b> {v}" for k,v in cluster_profiles.loc[c].items()])}
        </div>
        """, unsafe_allow_html=True)

    # Cluster Comparison Tool
    st.subheader("Compare Clusters")
    clust1 = st.selectbox("Select Cluster 1", cluster_profiles.index, key="c1")
    clust2 = st.selectbox("Select Cluster 2", cluster_profiles.index, key="c2")
    st.write("**Difference (Cluster 1 - Cluster 2):**")
    st.write((cluster_profiles.loc[clust1] - cluster_profiles.loc[clust2]).to_frame('Difference'))

    st.download_button("Download Cluster Data", df.to_csv(index=False).encode(), "clusters.csv")

# 4. Association Tab
elif page=="Association":
    st.header("🔗 Association Rules")
    cols=[c for c in df.columns if c.startswith(("Flavour_","Context_"))]
    s=st.slider("Min Support",0.01,0.2,0.05); c=st.slider("Min Confidence",0.1,0.7,0.3)
    freq=apriori(df[cols],min_support=s,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=c)
    rules["rule"]=rules["antecedents"].apply(lambda x:", ".join(x))+" → "+rules["consequents"].apply(lambda x:", ".join(x))
    top=rules.sort_values("lift",ascending=False).head(10)
    fig=px.bar(top,x="lift",y="rule",orientation="h", color="confidence", color_continuous_scale="Plasma", template=plotly_template)
    st.plotly_chart(fig,use_container_width=True)
    if not top.empty:
        top_rule = top.iloc[0]["rule"]
        st.markdown(f"**Insight:** Strongest association is: {top_rule}. Bundle these features/flavours in marketing.")
    # Network Graph
    if not top.empty:
        def association_graph(rules):
            G = nx.Graph()
            for _, row in rules.iterrows():
                for ante in row['antecedents']:
                    for cons in row['consequents']:
                        G.add_edge(ante, cons, weight=row['lift'])
            return G
        G = association_graph(top)
        fig_net = go.Figure()
        for edge in G.edges(data=True):
            fig_net.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode='lines+markers+text',
                text=[edge[0], edge[1]], textposition='top center'))
        fig_net.update_layout(title="Association Rule Network", template=plotly_template)
        st.plotly_chart(fig_net)

# 5. Anomaly Detection Tab (no change needed)

elif page=="Anomaly":
    st.header("🚨 Anomaly Detection")
    feats=["Age","MonthlyDisposableIncome","SpendPerServing","HealthConsciousness"]
    iso=IsolationForest(contamination=0.05,random_state=42).fit(df[feats])
    df["Anomaly"]=iso.predict(df[feats])
    fig=px.scatter(df,x="MonthlyDisposableIncome",y="SpendPerServing",
                   color=df["Anomaly"].map({1:"Normal",-1:"Anomaly"}),
                   color_discrete_sequence=["#00FF00","#FF0000"], template=plotly_template)
    st.plotly_chart(fig,use_container_width=True)
    outliers = (df["Anomaly"]==-1).sum()
    st.markdown(f"**Insight:** {outliers} anomalies detected. Review these cases for potential data errors or valuable edge customer behaviors.")

# 6. Regression Comparison Tab (unchanged, can add feature sliders if wanted)

elif page=="Regression":
    st.header("📈 Regression Comparison")
    Xr=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]; yr=df["SpendPerServing"]
    Xt,Xe,yt,ye=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    models={"Linear":LinearRegression(),"Lasso":Lasso(),"Ridge":Ridge(),"Tree":DecisionTreeRegressor()}
    res=[]
    for name,mdl in models.items():
        mdl.fit(Xt,yt); pr=mdl.predict(Xe)
        res.append({"Model":name,"R2":r2_score(ye,pr),"RMSE":np.sqrt(mean_squared_error(ye,pr))})
    rdf=pd.DataFrame(res)
    st.table(rdf)
    fig=px.bar(rdf,x="Model",y=["R2","RMSE"],barmode="group",
               color_discrete_sequence=px.colors.qualitative.Plotly, template=plotly_template)
    st.plotly_chart(fig,use_container_width=True)
    best = rdf.iloc[rdf['R2'].idxmax()]['Model']
    st.markdown(f"**Insight:** {best} regression gives the best fit for spend prediction. Use this to model expected sales for new cohorts.")

# 7. Forecasting Tab (unchanged)

elif page=="Forecasting":
    st.header("⏱️ Forecasting")
    ts=df[["SurveyDate","SpendPerServing"]].rename(columns={"SurveyDate":"ds","SpendPerServing":"y"}).dropna()
    if len(ts)>30:
        m=Prophet(); m.fit(ts); fut=m.make_future_dataframe(periods=30); fc=m.predict(fut)
        st.plotly_chart(plot_plotly(m,fc),use_container_width=True)
        next_week = fc.iloc[-7:]['yhat'].mean()
        st.markdown(f"**Insight:** Expected avg spend next week: AED {next_week:.2f}. Use for inventory planning.")
    else:
        st.warning("Not enough data for time series forecast.")

# 8. Cohort Tab
elif page=="Cohort":
    st.header("👥 Cohort Analysis")
    df["Cohort"]=df.SurveyDate.dt.to_period("M").astype(str)
    cr=df.groupby("Cohort")["SubscribePlan"].apply(lambda x:(x=="Yes").mean()).reset_index(name="Rate")
    fig=px.line(cr,x="Cohort",y="Rate",markers=True, template=plotly_template); st.plotly_chart(fig,use_container_width=True)
    most = cr.sort_values("Rate", ascending=False).iloc[0]
    st.markdown(f"**Insight:** {most['Cohort']} cohort had highest subscription rate. Time campaigns with these patterns.")

    # Retention heatmap
    if "SubscribePlan" in df.columns:
        pivot = df.pivot_table(index=df['Cohort'], columns=df['SurveyDate'].dt.month, 
            values='SubscribePlan', aggfunc=lambda x: (x=="Yes").mean())
        fig_ret = px.imshow(pivot, color_continuous_scale='Blues', 
            labels=dict(color="Retention Rate"), template=plotly_template)
        st.plotly_chart(fig_ret, use_container_width=True)

# 9. Geography Tab (unchanged)

elif page=="Geography":
    st.header("🗺️ Average Spend Per City")
    if "City" in df.columns and not df["City"].isnull().all():
        city_stats = df.groupby("City")["SpendPerServing"].mean().reset_index()
        city_stats = city_stats.sort_values("SpendPerServing", ascending=False)
        fig = px.bar(
            city_stats, x="City", y="SpendPerServing",
            color="SpendPerServing",
            color_continuous_scale="Viridis",
            labels={"SpendPerServing": "Avg Spend (AED)"},
            title="Average Spend by City",
            template=plotly_template
        )
        st.plotly_chart(fig, use_container_width=True)
        overall_avg = df["SpendPerServing"].mean()
        top_city = city_stats.iloc[0]
        low_city = city_stats.iloc[-1]
        pct_top = ((top_city["SpendPerServing"] - overall_avg) / overall_avg) * 100
        pct_low = ((low_city["SpendPerServing"] - overall_avg) / overall_avg) * 100

        insight = (
            f"**Insight:**\n"
            f"- **{top_city['City']}** has the highest average spend per serving: **AED {top_city['SpendPerServing']:.2f}** "
            f"({pct_top:+.1f}% vs. UAE avg).\n"
            f"- **{low_city['City']}** has the lowest at **AED {low_city['SpendPerServing']:.2f}** ({pct_low:+.1f}% vs. avg).\n"
        )
        if pct_top > 8:
            insight += (
                f"> **Business Implication:** Focus premium pricing, exclusive launches, or loyalty programs in {top_city['City']}. "
                f"For price-sensitive cities like {low_city['City']}, test value packs or discount offers."
            )
        elif pct_low < -8:
            insight += (
                f"> **Business Implication:** Consider tailoring promotions for {low_city['City']} to boost trial and repeat purchases, "
                f"while leveraging {top_city['City']} for higher-margin SKUs."
            )
        else:
            insight += (
                f"> **Business Implication:** Spending is similar across cities—consider citywide marketing with modest regional tweaks."
            )
        st.markdown(insight)
    else:
        st.warning("No city data available in this dataset.")

# 10. Sentiment Tab
elif page=="Sentiment":
    st.header("💬 Sentiment Word Cloud")
    text=" ".join(df.Feedback.astype(str))
    wc=WordCloud(width=800,height=400).generate(text)
    fig,ax=plt.subplots(figsize=(10,5)); ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig)

    # Sentiment Score Pie
    sent = df['Feedback'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    sentiment = pd.cut(sent, bins=[-1,-0.1,0.1,1], labels=['Negative','Neutral','Positive'])
    sent_count = sentiment.value_counts()
    fig_sent = px.pie(names=sent_count.index, values=sent_count.values, title="Feedback Sentiment", template=plotly_template)
    st.plotly_chart(fig_sent, use_container_width=True)
    st.markdown(f"**Insight:** {sent_count.idxmax()} feedback is most common.")

    # Top Comments
    st.subheader("Sample Comments")
    pos = df.loc[sent > 0.1, 'Feedback'].dropna().sample(min(3, (sent > 0.1).sum()))
    neg = df.loc[sent < -0.1, 'Feedback'].dropna().sample(min(3, (sent < -0.1).sum()))
    st.write("**Positive:**")
    for i in pos: st.markdown(f"- {i}")
    st.write("**Negative:**")
    for i in neg: st.markdown(f"- {i}")

# 11. LTV & Churn Tab
elif page=="LTV & Churn":
    st.header("💰 LTV & Churn")
    df["FreqNum"]=df.ConsumptionFrequency.map({"Never":0,"Rarely":1,"1-2":2,"3-4":4,"5+":5})
    df["LTV"]=df.SpendPerServing*df.FreqNum*12
    churn=(df.SubscribePlan=="No").astype(int)
    Xc=df[["MonthlyDisposableIncome","HealthConsciousness","Age"]]
    Xt,Xe,yt,ye=train_test_split(Xc,churn,test_size=0.2,random_state=42)
    clf=RandomForestClassifier(random_state=42).fit(Xt,yt); pr=clf.predict(Xe)
    metrics={"Accuracy":accuracy_score(ye,pr),"Precision":precision_score(ye,pr),"Recall":recall_score(ye,pr)}
    st.table(pd.DataFrame.from_dict(metrics,orient='index',columns=['Value']))
    st.plotly_chart(px.histogram(df,x="LTV",nbins=30, template=plotly_template),use_container_width=True)
    avg_ltv = df["LTV"].mean()
    st.markdown(f"**Insight:** Average LTV is AED {avg_ltv:.2f}. Focus retention efforts on high-LTV, high-churn-risk segments.")
    # Churn probability distribution
    churn_prob = clf.predict_proba(Xe)[:,1]
    fig_churn = px.histogram(churn_prob, nbins=20, title="Predicted Churn Probability", template=plotly_template)
    st.plotly_chart(fig_churn, use_container_width=True)
    st.markdown("**Insight:** Most users have low churn risk, but focus on those above 0.7.")

# 12. Price Elasticity Tab (unchanged)

elif page=="Price Elasticity":
    st.header("💵 Price Elasticity")
    price=st.slider("Price per Serving",5,30,12)
    buyers=df[df.SpendPerServing>=price].shape[0]
    revenue=buyers*price
    st.metric("Buyers",buyers); st.metric("Revenue (AED)",revenue)
    st.markdown("**Insight:** As price increases, buyer count drops but revenue can peak at optimal price. Use this to set prices.")

# 13. Glossary Tab
elif page=="Glossary":
    st.header("📚 Data Glossary")
    dictionary = {
        "SpendPerServing": "How much a customer spends for each serving of the health drink.",
        "HealthConsciousness": "How important health is to the consumer (scale 1-10).",
        "TryNewBrand": "Whether the customer is willing to try a new brand (Yes/No).",
        "SubscribePlan": "If the customer has opted for a subscription plan.",
        "MonthlyDisposableIncome": "Monthly spending money available to the customer.",
        "Age": "Age of respondent.",
        "ExerciseFrequency": "How often the person exercises.",
        "ConsumptionFrequency": "How often the health drink is consumed.",
        "Feedback": "Free text user feedback or comments."
    }
    for k, v in dictionary.items():
        st.markdown(f"**{k}**: {v}")

if st.button("💡 Send Feedback"):
    fb=st.text_area("Your feedback:")
    if st.button("Submit"):
        with open("feedback.txt","a") as f: f.write(fb+"\\n---\\n")
        st.success("Thank you!")

