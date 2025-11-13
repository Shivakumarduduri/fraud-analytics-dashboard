import pandas as pd
import zipfile
import os

# ------------------------------------
# SAFE ZIP EXTRACTION
# ------------------------------------
def extract_zip(uploaded_file):
    extract_path = "extracted_data"
    os.makedirs(extract_path, exist_ok=True)

    # Save uploaded ZIP temporarily
    temp_zip_path = os.path.join(extract_path, "uploaded.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract ZIP
    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path


# ------------------------------------
# FIND ALL CSV FILES IN ZIP
# ------------------------------------
def find_csv_files(extract_path):
    csv_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


# ------------------------------------
# LOAD CSVs AND DETECT WHICH ONE IS TRANS / CUSTOMER / FRAUD
# ------------------------------------
def load_csvs(extract_path):
    csv_files = find_csv_files(extract_path)

    trans_file, cust_file, fraud_file = None, None, None

    for path in csv_files:
        df = pd.read_csv(path)
        file = os.path.basename(path).lower()

        # Detect transaction file
        if any(key in file for key in ["trans", "transaction", "tx"]):
            trans_file = df

        # Detect customer file
        elif any(key in file for key in ["cust", "customer", "client"]):
            cust_file = df

        # Detect fraud label file
        elif any(key in file for key in ["fraud", "label", "class"]):
            fraud_file = df

        else:
            # Default fallback: first file is treated as transaction file
            if trans_file is None:
                trans_file = df

    return trans_file, cust_file, fraud_file


# ----------------------------------------------------
# FIX COLUMN NAME DETECTION (MULTIPLE FORMAT SUPPORT)
# ----------------------------------------------------
def find_column(df, possible_names):
    """
    Returns the first matching column from a list of possible names.
    """
    df_cols_lower = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_cols_lower:
            return df_cols_lower[name.lower()]
    return None


# ------------------------------------
# MERGE LOGIC WITH SAFETY CHECKS
# ------------------------------------
def merge_data(trans_file, cust_file=None, fraud_file=None):
    if trans_file is None:
        return pd.DataFrame()

    df = trans_file.copy()

    # ---- Identify key columns ----
    trans_id = find_column(df, ["TransactionID", "transactionid", "trans_id", "txid", "id"])
    cust_id = find_column(df, ["CustomerID", "customerid", "cust_id", "client_id"])

    # ---------------------------
    # MERGE CUSTOMER FILE
    # ---------------------------
    if cust_file is not None:
        cust_id_file = find_column(cust_file, ["CustomerID", "customerid", "cust_id", "client_id"])

        if cust_id and cust_id_file:
            df = df.merge(cust_file, left_on=cust_id, right_on=cust_id_file, how="left")
        else:
            print("⚠️ CustomerID mismatch — skipping customer merge.")

    # ---------------------------
    # MERGE FRAUD LABEL FILE
    # ---------------------------
    if fraud_file is not None:
        fraud_trans_id = find_column(fraud_file, ["TransactionID", "transactionid", "trans_id", "txid", "id"])

        if trans_id and fraud_trans_id:
            df = df.merge(fraud_file, left_on=trans_id, right_on=fraud_trans_id, how="left")
        else:
            print("⚠️ TransactionID mismatch — skipping fraud merge.")

    return df
