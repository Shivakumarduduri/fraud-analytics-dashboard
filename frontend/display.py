import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix

def display_table(df, title):
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

def display_accuracy(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    st.write(f"**{model_name} Accuracy:** {acc:.2f}%")
    st.text(f"Confusion Matrix:\n{cm}")

def show_summary_charts(df, label_col="FraudIndicator"):
    if label_col not in df.columns:
        st.warning(f"No '{label_col}' column found for analysis.")
        return

    fig = px.histogram(df, x="Amount", color=label_col,
                       title="Transaction Amount Distribution by Fraud Status")
    st.plotly_chart(fig, use_container_width=True)

def download_button(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")
