import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

st.title("📈 Auto Plots")

def generate_download_link(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="plot.png">📥 Download Plot as PNG</a>'
    return href

if "shared_df" in st.session_state:
    df = st.session_state.df

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    plot_type = st.sidebar.selectbox("Choose a plot type", ["Histogram", "Boxplot", "Correlation Heatmap", "Countplot", "Missing Values Heatmap"])

    if plot_type == "Histogram":
        col = st.selectbox("Choose a numeric column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
        st.markdown(generate_download_link(fig), unsafe_allow_html=True)

    elif plot_type == "Boxplot":
        col = st.selectbox("Choose a numeric column", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)
        st.markdown(generate_download_link(fig), unsafe_allow_html=True)

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.markdown(generate_download_link(fig), unsafe_allow_html=True)

    elif plot_type == "Countplot":
        col = st.selectbox("Choose a categorical column", cat_cols)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
        st.pyplot(fig)
        st.markdown(generate_download_link(fig), unsafe_allow_html=True)

    elif plot_type == "Missing Values Heatmap":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
        st.markdown(generate_download_link(fig), unsafe_allow_html=True)
else:
    st.warning("📂 Please upload a dataset from the Home page.")


