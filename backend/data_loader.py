import pandas as pd
import zipfile
import os

def extract_zip(uploaded_file):
    extract_path = "extracted_data"
    os.makedirs(extract_path, exist_ok=True)

    # Save uploaded zip temporarily
    temp_zip_path = os.path.join(extract_path, "uploaded.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract all
    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path


def find_csv_files(extract_path):
    csv_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def load_csvs(extract_path):
    csv_files = find_csv_files(extract_path)
    trans_file, cust_file, fraud_file = None, None, None

    for path in csv_files:
        file = os.path.basename(path).lower()
        df = pd.read_csv(path)

        if "trans" in file:
            trans_file = df
        elif "cust" in file:
            cust_file = df
        elif "fraud" in file:
            fraud_file = df
        else:
            # default fallback if only one CSV exists
            if trans_file is None:
                trans_file = df

    return trans_file, cust_file, fraud_file


def merge_data(trans_file, cust_file=None, fraud_file=None):
    if trans_file is None:
        return pd.DataFrame()

    df = trans_file.copy()
    if cust_file is not None:
        df = df.merge(cust_file, on="CustomerID", how="left")
    if fraud_file is not None:
        df = df.merge(fraud_file, on="TransactionID", how="left")
    return df
