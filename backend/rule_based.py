import pandas as pd

def detect_rule_based(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Auto-detect amount column
    amount_col = None
    for c in df.columns:
        if "amount" in c.lower() or "amt" in c.lower():
            amount_col = c
            break

    if not amount_col:
        df["Fraudulent_Rule"] = False
        return df

    # Apply rule conditions
    median_amt = df[amount_col].median()
    mean_amt = df[amount_col].mean()

    df["Fraudulent_Rule"] = (
        (df[amount_col] > mean_amt * 2)
        | (df[amount_col] > median_amt * 2)
        | (df[amount_col] > 1000)  # threshold safety
    )

    # Extra rule — if other columns present
    if "CustomerID" in df.columns:
        freq = df["CustomerID"].value_counts()
        risky_customers = freq[freq > 5].index
        df.loc[df["CustomerID"].isin(risky_customers), "Fraudulent_Rule"] = True

    return df
