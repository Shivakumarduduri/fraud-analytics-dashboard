import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from backend.rule_based import detect_rule_based
from backend.data_loader import extract_zip, load_csvs, merge_data
from backend.db_handler import get_db_connection
from frontend.display import (
    display_table, download_button,
    display_accuracy, show_summary_charts
)

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Fraud Analytics Dashboard", layout="wide", page_icon="💳")

# ==================== STYLE ====================
st.markdown("""
<style>
.stApp {background-color:#111827;color:#e5e7eb;font-family:'Poppins',sans-serif;}
section[data-testid="stSidebar"]{background-color:#1f2937;color:white;padding-top:25px;border-right:1px solid #374151;}
h1,h2,h3,h4{color:white!important;}
div[data-testid="stDataFrame"]{background-color:#1e293b;border-radius:10px;padding:10px;}
.metric {color:white!important;}
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.title("💳 Fraud Analytics Dashboard")

# ==================== SIDEBAR ====================
st.sidebar.markdown("### 💳 Input Type")
input_type = st.sidebar.selectbox("Select Input Type", ["Select", "Upload ZIP", "MongoDB Database"])

st.sidebar.markdown("### 🧠 Method / Algorithm")
method = st.sidebar.selectbox(
    "Select Method",
    ["Select", "Overview", "Rule-based Detection", "ML Models", "Analytics", "Combined Fraud"]
)

st.sidebar.markdown("### ⚡ Real-time Analysis")
realtime_analysis = st.sidebar.selectbox("Real-time Options", ["Select", "Real-time Fraud Transaction Check"])

df = None

# ==================== HELPERS ====================
def normalize_label(df):
    """Ensure consistent fraud label column."""
    for c in df.columns:
        if c.lower() in ["fraudindicator", "class", "isfraud", "fraud"]:
            df.rename(columns={c: "FraudIndicator"}, inplace=True)
            return df
    st.warning("⚠️ No fraud label found. Creating synthetic 'FraudIndicator'.")
    df["FraudIndicator"] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    return df


def only_numeric(df):
    """Keep only numeric columns for ML."""
    return df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()


def run_ml_show_all(df):
    """Run Random Forest, Logistic Regression, and Isolation Forest."""
    st.subheader("🤖 Machine Learning Detection")

    if "FraudIndicator" not in df.columns:
        st.error("No 'FraudIndicator' column found.")
        return

    if df["FraudIndicator"].nunique() < 2:
        st.warning("Dataset lacks both fraud & non-fraud samples.")
        return

    df = only_numeric(df)
    X = df.drop("FraudIndicator", axis=1)
    y = df["FraudIndicator"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---- RANDOM FOREST ----
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    df_rf = X_test.copy()
    df_rf["Fraudulent_RF"] = y_pred_rf

    st.markdown("### 🌲 Random Forest Fraud Predictions")
    display_table(df_rf, "Random Forest Fraud Predictions")
    st.write(f"**Random Forest Accuracy:** {acc_rf*100:.2f}%")
    st.write("Confusion Matrix:")
    st.write(cm_rf)

    # ---- LOGISTIC REGRESSION ----
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    df_lr = X_test.copy()
    df_lr["Fraudulent_LR"] = y_pred_lr

    st.markdown("### 📊 Logistic Regression Fraud Predictions")
    display_table(df_lr, "Logistic Regression Fraud Predictions")
    st.write(f"**Logistic Regression Accuracy:** {acc_lr*100:.2f}%")
    st.write("Confusion Matrix:")
    st.write(cm_lr)

    # ---- ISOLATION FOREST ----
    iso = IsolationForest(contamination=0.1, random_state=42)
    y_pred_iso = iso.fit_predict(X_test)
    y_pred_iso = np.where(y_pred_iso == -1, 1, 0)
    acc_iso = accuracy_score(y_test, y_pred_iso)
    cm_iso = confusion_matrix(y_test, y_pred_iso)
    df_iso = X_test.copy()
    df_iso["Fraudulent_ISO"] = y_pred_iso

    st.markdown("### 🧩 Isolation Forest Fraud Predictions")
    display_table(df_iso, "Isolation Forest Fraud Predictions")
    st.write(f"**Isolation Forest Accuracy:** {acc_iso*100:.2f}%")
    st.write("Confusion Matrix:")
    st.write(cm_iso)

# ==================== REAL-TIME TRANSACTION CHECK ====================
if realtime_analysis == "Real-time Fraud Transaction Check":
    st.subheader("⚡ Real-time Fraud Transaction Check")
    st.markdown("Enter transaction details manually:")

    tx = {}
    tx["amount"] = st.number_input("Transaction Amount", min_value=0.0, value=120.0)
    tx["hour"] = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
    tx["customer_age"] = st.number_input("Customer Age", 18, 100, 35)
    tx["distance_from_home_km"] = st.number_input("Distance from Home (km)", 0.0, 2000.0, 10.0)
    tx["num_prev_24h"] = st.number_input("Transactions in last 24h", 0, 100, 1)
    tx["avg_amount_30d"] = st.number_input("Avg Transaction Amount (30d)", 0.0, 10000.0, 500.0)
    tx["is_foreign"] = st.selectbox("Is Foreign Transaction?", [0, 1])
    tx["card_present"] = st.selectbox("Card Present?", [1, 0])
    tx["device_trusted"] = st.selectbox("Trusted Device?", [1, 0])
    tx["merchant_risk_score"] = st.slider("Merchant Risk Score", 0.0, 1.0, 0.2)

    def rule_based(tx):
        score, reasons = 0.0, []
        if tx["amount"] > 2000:
            score += 0.35; reasons.append("very large amount")
        elif tx["amount"] > 500:
            score += 0.12; reasons.append("large amount")

        if tx["distance_from_home_km"] > 500:
            score += 0.20; reasons.append("transaction far from home")
        elif tx["distance_from_home_km"] > 100:
            score += 0.07

        if tx["is_foreign"]:
            score += 0.25; reasons.append("foreign transaction")
        if tx["num_prev_24h"] > 20:
            score += 0.18; reasons.append("too many transactions")
        if not tx["device_trusted"]:
            score += 0.07; reasons.append("untrusted device")
        if not tx["card_present"]:
            score += 0.06; reasons.append("card-not-present")
        score += tx["merchant_risk_score"] * 0.2
        if tx["merchant_risk_score"] > 0.6:
            reasons.append("high merchant risk")
        if tx["hour"] < 6 or tx["hour"] > 23:
            score += 0.03

        decision = "Fraud" if score >= 0.35 else "Legit"
        return {"decision": decision, "score": score, "reasons": reasons}

    def rf_predict(tx):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        df_train = pd.DataFrame({
            "amount": np.random.uniform(10, 5000, 5000),
            "hour": np.random.randint(0, 24, 5000),
            "customer_age": np.random.randint(18, 70, 5000),
            "distance_from_home_km": np.random.uniform(0, 1000, 5000),
            "num_prev_24h": np.random.poisson(2, 5000),
            "avg_amount_30d": np.random.uniform(10, 1000, 5000),
            "is_foreign": np.random.binomial(1, 0.05, 5000),
            "card_present": np.random.binomial(1, 0.8, 5000),
            "device_trusted": np.random.binomial(1, 0.85, 5000),
            "merchant_risk_score": np.random.rand(5000),
        })
        y = np.random.binomial(1, 0.1, 5000)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_train)
        model.fit(X_scaled, y)
        X = pd.DataFrame([tx])
        X_scaled_tx = scaler.transform(X)
        prob = model.predict_proba(X_scaled_tx)[0][1]
        decision = "Fraud" if prob >= 0.5 else "Legit"
        return {"decision": decision, "prob_fraud": prob}

    if st.button("🔍 Predict Fraud"):
        rule = rule_based(tx)
        rf = rf_predict(tx)
        combined_score = 0.6 * rf["prob_fraud"] + 0.4 * rule["score"]
        combined = "Fraud" if combined_score >= 0.4 else "Legit"

        st.markdown("### Results")
        st.write(f"**Rule-based:** {rule['decision']} (score={rule['score']:.2f})")
        if rule["reasons"]:
            st.write("Reasons:", ", ".join(rule["reasons"]))
        st.write(f"**Random Forest:** {rf['decision']} (fraud_prob={rf['prob_fraud']:.2f})")

        if combined == "Fraud":
            st.markdown("<h3 style='color:red;font-weight:bold;'>🚨 COMBINED DECISION: FRAUD</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:green;font-weight:bold;'>✅ COMBINED DECISION: LEGIT</h3>", unsafe_allow_html=True)
        st.write(f"(combined_score={combined_score:.2f})")

# ==================== MONGODB DATABASE MODE ====================
elif input_type == "MongoDB Database":
    db = get_db_connection()
    if db is not None:
        collections = db.list_collection_names()
        if not collections:
            st.error("❌ No collections found in MongoDB database.")
        else:
            default_col = "transactions" if "transactions" in collections else collections[0]
            df = pd.DataFrame(list(db[default_col].find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df = normalize_label(df)
            st.success(f"✅ Connected and loaded `{default_col}` automatically!")

# ==================== ZIP UPLOAD MODE ====================
elif input_type == "Upload ZIP":
    uploaded_file = st.sidebar.file_uploader("Upload ZIP File", type=["zip"])
    if uploaded_file:
        st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")
        extract_path = extract_zip(uploaded_file)
        trans_file, cust_file, fraud_file = load_csvs(extract_path)
        if trans_file is not None:
            df = merge_data(trans_file, cust_file, fraud_file)
            df = normalize_label(df)
            st.success("✅ Data extracted and merged successfully!")

# ==================== METHOD EXECUTION ====================
if df is not None and method != "Select":
    # ---------- OVERVIEW ----------
    if method == "Overview":
        st.subheader("📂 Overview Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("Unique Customers", df.CustomerID.nunique() if "CustomerID" in df.columns else "N/A")
        col3.metric("Fraudulent Transactions", int(df["FraudIndicator"].sum()))

        fraud_df = df[df["FraudIndicator"] == 1]

        if fraud_df.empty:
            st.warning("✅ No fraudulent transactions detected in this dataset!")
        else:
            st.markdown("### 🚨 Fraudulent Transactions Table")
            display_table(fraud_df, "Fraudulent Transactions (All)")
            download_button(fraud_df, "fraud_overview.csv", "Download Fraud Transactions")

            if "Amount" in fraud_df.columns:
                fig = px.histogram(
                    fraud_df,
                    x="Amount",
                    nbins=20,
                    color_discrete_sequence=["#ff4b4b"],
                    title="Fraud Transaction Amount Distribution"
                )
                st.plotly_chart(fig, width='stretch')

    # ---------- RULE-BASED DETECTION ----------
    elif method == "Rule-based Detection":
        st.subheader("🚨 Rule-based Fraud Detection")
        df_rule = detect_rule_based(df.copy())
        fraud_df_rule = df_rule[df_rule["Fraudulent_Rule"] == True]
        if fraud_df_rule.empty:
            st.warning("No frauds detected by rule-based method.")
        else:
            display_table(fraud_df_rule, "Rule-based Fraud Transactions")
            download_button(fraud_df_rule, "fraud_rule.csv", "Download Rule-based CSV")
            if "Amount" in fraud_df_rule.columns:
                fig = px.histogram(
                    fraud_df_rule, x="Amount", nbins=20,
                    color_discrete_sequence=["#ff4b4b"],
                    title="Fraud Amount Distribution"
                )
                st.plotly_chart(fig, width='stretch')

    # ---------- ML MODELS ----------
    elif method == "ML Models":
        run_ml_show_all(df.copy())

    # ---------- ANALYTICS ----------
    elif method == "Analytics":
        st.subheader("📈 Analytics & Visualizations")
        show_summary_charts(df, label_col="FraudIndicator")

    # ---------- COMBINED FRAUD ----------
    elif method == "Combined Fraud":
        st.subheader("🧩 Combined Rule-based + Random Forest Fraud")
        df_rule = detect_rule_based(df.copy())
        df_num = only_numeric(df.copy())

        if "FraudIndicator" in df_num.columns and df_num["FraudIndicator"].nunique() >= 2:
            X = df_num.drop("FraudIndicator", axis=1)
            y = df_num["FraudIndicator"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            df_combined = X_test.copy()
            df_combined["Fraudulent_RF"] = y_pred_rf
            df_combined["Fraudulent_Rule"] = detect_rule_based(df.copy()).get("Fraudulent_Rule", False)
            df_combined["Combined_Fraud"] = np.where(
                (df_combined["Fraudulent_RF"] == 1) & (df_combined["Fraudulent_Rule"] == True),
                "Both",
                np.where(df_combined["Fraudulent_Rule"] == True, "Rule Only",
                         np.where(df_combined["Fraudulent_RF"] == 1, "RF Only", "None"))
            )
            fraud_combined = df_combined[df_combined["Combined_Fraud"] != "None"]
            display_table(fraud_combined, "Combined Fraud Results")
            download_button(fraud_combined, "combined_fraud.csv", "Download Combined Fraud CSV")

            if "Amount" in df_combined.columns:
                fig_comb = px.histogram(
                    fraud_combined,
                    x="Amount", color="Combined_Fraud",
                    title="Combined Fraud Distribution"
                )
                st.plotly_chart(fig_comb, width='stretch')
        else:
            st.warning("Not enough labeled data to run Combined Fraud analysis.")

else:
    st.info("Select an Input Type, Method, or Real-time Analysis to get started.")
