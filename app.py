import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
    mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide", page_title="Health Drink Dashboard")
st.title("🚀 Health Drink Survey Dashboard")

# --- Sidebar: Data Source ---
st.sidebar.header("Data Source")
data_option = st.sidebar.radio("Choose source:", ["GitHub URL", "Upload CSV"])
if data_option == "GitHub URL":
    default_url = "https://raw.githubusercontent.com/sagittarius251201/new-app/refs/heads/main/health_drink_survey_1000_responses%20(1).csv"
    url = st.sidebar.text_input("Raw GitHub CSV URL", value=default_url)
    df = pd.read_csv(url)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a CSV.")
        st.stop()

# --- Tabs ---
tabs = st.tabs([
    "1️⃣ Data Visualization", 
    "2️⃣ Classification", 
    "3️⃣ Clustering", 
    "4️⃣ Association Mining", 
    "5️⃣ Regression"
])

# --- 1. Data Viz ---
with tabs[0]:
    st.header("Data Visualization")
    # 1. Age dist
    fig, ax = plt.subplots()
    ax.hist(df['Age'], bins=20)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    # 2. Income dist
    fig, ax = plt.subplots()
    ax.hist(df['MonthlyDisposableIncome'], bins=20)
    ax.set_title("Income Distribution (AED)")
    st.pyplot(fig)

    # 3. Spend per serving
    fig, ax = plt.subplots()
    ax.hist(df['SpendPerServing'], bins=20)
    ax.set_title("Spend per Serving (AED)")
    st.pyplot(fig)

    # 4. Gender vs Spend
    fig, ax = plt.subplots()
    df.boxplot(column='SpendPerServing', by='Gender', ax=ax)
    ax.set_title("Spend by Gender")
    st.pyplot(fig)

    # 5. Occupation counts
    fig, ax = plt.subplots()
    df['Occupation'].value_counts().plot.bar(ax=ax)
    ax.set_title("Occupation Counts")
    st.pyplot(fig)

    # 6. Health consciousness
    fig, ax = plt.subplots()
    df['HealthConsciousness'].value_counts(sort=False).plot.bar(ax=ax)
    ax.set_title("Health Consciousness Levels")
    st.pyplot(fig)

    # 7. Consumption freq
    fig, ax = plt.subplots()
    df['ConsumptionFrequency'].value_counts().plot.bar(ax=ax)
    ax.set_title("Consumption Frequency")
    st.pyplot(fig)

    # 8. Top benefits
    fig, ax = plt.subplots()
    df['TopHealthBenefit'].value_counts().plot.bar(ax=ax)
    ax.set_title("Top Health Benefits")
    st.pyplot(fig)

    # 9. Packaging preferences
    fig, ax = plt.subplots()
    df['PackagingFormat'].value_counts().plot.bar(ax=ax)
    ax.set_title("Packaging Format")
    st.pyplot(fig)

    # 10. Correlation heatmap
    num_df = df.select_dtypes(include=['int64','float64'])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)

# --- 2. Classification ---
with tabs[1]:
    st.header("Classification (TryNewBrand)")
    X = df.select_dtypes(include=['int64','float64']).drop(columns=['SpendPerServing'])
    y = LabelEncoder().fit_transform(df['TryNewBrand'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_clf = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }
    metrics = []
    cm_data = {}
    roc_data = {}

    for name, model in models_clf.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        metrics.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
        cm_data[name] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

    st.subheader("Metrics Comparison")
    st.dataframe(pd.DataFrame(metrics))

    sel = st.selectbox("Confusion Matrix for:", list(models_clf.keys()))
    cm = cm_data[sel]
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for (i,j),val in np.ndenumerate(cm):
        ax.text(j,i,val,ha='center')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name,(fpr,tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=name)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    st.pyplot(fig)

    st.subheader("Predict New Data")
    upload = st.file_uploader("Upload CSV (no target)", type="csv")
    if upload:
        new_df = pd.read_csv(upload)
        preds = models_clf[sel].predict(new_df[X.columns])
        new_df['TryNewBrand_Pred'] = LabelEncoder().inverse_transform(preds)
        st.dataframe(new_df)
        st.download_button("Download Predictions", new_df.to_csv(index=False).encode(), "preds.csv")

# --- 3. Clustering ---
with tabs[2]:
    st.header("Clustering (K-Means)")
    feats = ['Age','MonthlyDisposableIncome','SpendPerServing','HealthConsciousness']
    data = df[feats]
    st.subheader("Elbow Chart")
    sse = [KMeans(n_clusters=k, random_state=42).fit(data).inertia_ for k in range(1,11)]
    fig, ax = plt.subplots()
    ax.plot(range(1,11), sse, marker='o')
    st.pyplot(fig)

    k = st.slider("Clusters (k)", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    df['Cluster'] = km.labels_
    st.subheader("Cluster Personas")
    st.dataframe(df.groupby('Cluster')[feats].mean().round(2))
    st.download_button("Download Clustered Data", df.to_csv(index=False).encode(), "clustered.csv")

# --- 4. Association Mining ---
with tabs[3]:
    st.header("Association Rule Mining")
    cols = [c for c in df.columns if ('Flavour_' in c) or ('Context_' in c)]
    basket = df[cols]
    min_sup = st.slider("Min Support", 0.01, 0.1, 0.02)
    min_conf = st.slider("Min Confidence", 0.1, 0.5, 0.3)
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    st.subheader("Top-10 Rules by Lift")
    st.dataframe(rules.sort_values('lift', ascending=False).head(10))

# --- 5. Regression ---
with tabs[4]:
    st.header("Regression on SpendPerServing")
    Xr = df[['MonthlyDisposableIncome','HealthConsciousness','Age']]
    yr = df['SpendPerServing']
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    models_reg = {
        'Linear': LinearRegression(),


    Ridge': Ridge(),
        'Lasso': Lasso(),
        'DT Regressor': DecisionTreeRegressor(random_state=42)
    }
    reg_res = []
    for name, mdl in models_reg.items():
        mdl.fit(Xtr, ytr)
        p_train = mdl.predict(Xtr); p_test = mdl.predict(Xte)
        reg_res.append({
            'Model': name,
            'Train R2': r2_score(ytr, p_train),
            'Test R2': r2_score(yte, p_test),
            'Train RMSE': mean_squared_error(ytr, p_train, squared=False),
            'Test RMSE': mean_squared_error(yte, p_test, squared=False)
        })
    st.subheader("Regression Results")
    st.dataframe(pd.DataFrame(reg_res))
