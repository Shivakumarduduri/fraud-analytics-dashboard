from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def supervised_ml(df):
    df = df.dropna()
    if "FraudIndicator" not in df.columns:
        df["FraudIndicator"] = (df["Amount"] > 2000).astype(int)

    X = df.select_dtypes(include=["number"]).drop(columns=["FraudIndicator"], errors="ignore")
    y = df["FraudIndicator"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    df_test = X_test.copy()
    df_test["Fraudulent_RF"] = y_pred_rf
    df_test["Fraudulent_LR"] = y_pred_lr

    return df_test, y_test, y_pred_rf, y_pred_lr

def unsupervised_iso(df):
    X = df.select_dtypes(include=["number"]).fillna(0)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["Fraudulent_ISO"] = iso.fit_predict(X)
    df["Fraudulent_ISO"] = df["Fraudulent_ISO"].apply(lambda x: 1 if x == -1 else 0)
    return df
