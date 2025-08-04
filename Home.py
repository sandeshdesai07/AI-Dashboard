import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import io
import base64
import requests

st.set_page_config(page_title="AI Analytics Dashboard", layout="wide")
st.title("📈 AI-Powered Analytics Dashboard")

# ---------- Styling ----------
st.markdown("""
    <style>
    .bordered-section {
        border: 2px solid #D3D3D3;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 1rem;
    }

    .stFileUploader {
        padding: 1rem;
        border-radius: 6px;
        background-color: #f9f9f9;
    }

    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    .floating-rectangle {
        position: absolute;
        top: 80px;
        left: 50px;
        width: 90%;
        height: 80vh;
        background-color: #f0f0f0;
        border: 2px solid #ccc;
        border-radius: 12px;
        z-index: -1;
    }
    </style>
    <div class="floating-rectangle"></div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- File Upload ----------
with st.container():
    st.markdown("### 🪄 Upload and Understand Your Data")
    st.write("Upload your dataset and explore its structure and basic statistics.")
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
#uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        st.session_state["shared_df"] = df
        st.session_state.df = df
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        st.session_state["shared_df"] = df
        st.session_state.df = df
    
    # Save DataFrame in session state
    st.session_state['shared_df'] = df
    st.success("✅ Dataset uploaded successfully!")

# Use previously uploaded file if user switches back to this page
elif 'shared_df' in st.session_state:
    df = st.session_state['shared_df']
    #st.info("📁 Using previously uploaded file.")
else:
    df = None
    #st.warning("📂 Please upload a dataset.")

# ---------- Download Function ----------
def generate_download_link(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="plot.png">📥 Download Plot as PNG</a>'
    return href

# ---------- Data and Plots ----------
if df is not None:
    with st.expander("🔍 Preview Uploaded Dataset", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📊 Summary Statistics"):
        st.write(df.describe())

    with st.expander("🧼 Missing Values Summary"):
        st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.subheader("📈 Auto Plots")
    st.sidebar.title("🛠️ Options Panel")
    plot_type = st.sidebar.selectbox("📊 Choose a Plot Type", [
    "Histogram", "Boxplot", "Correlation Heatmap", "Countplot",
    "Missing Values Heatmap", "Pairplot", "Violinplot"])
    
    if plot_type == "Histogram":
        col = st.selectbox("Choose a numeric column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
        #st.markdown(generate_download_link(fig), unsafe_allow_html=True)
        st.markdown(
    f'<div style="text-align: center;">{generate_download_link(fig)}</div>',
    unsafe_allow_html=True
)

    elif plot_type == "Boxplot":
        col = st.selectbox("Choose a numeric column", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)
        #st.markdown(generate_download_link(fig), unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center;">{generate_download_link(fig)}</div>',unsafe_allow_html=True)

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        #st.markdown(generate_download_link(fig), unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center;">{generate_download_link(fig)}</div>',unsafe_allow_html=True)

    elif plot_type == "Countplot":
        col = st.selectbox("Choose a categorical column", cat_cols)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
        st.pyplot(fig)
        #st.markdown(generate_download_link(fig), unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center;">{generate_download_link(fig)}</div>',unsafe_allow_html=True)

    elif plot_type == "Missing Values Heatmap":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
        #st.markdown(generate_download_link(fig), unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center;">{generate_download_link(fig)}</div>',unsafe_allow_html=True)

    elif plot_type == "Pairplot":
        st.info("📌 Generating pairplot. This may take a few seconds for large datasets.")
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)

    elif plot_type == "Violinplot":
        y_col = st.selectbox("Choose a numeric column for y-axis", numeric_cols)
        x_col = st.selectbox("Choose a categorical column for x-axis", cat_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x=x_col, y=y_col, data=df, ax=ax)
        st.pyplot(fig)
        #st.markdown(generate_download_link(fig), unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center;">{generate_download_link(fig)}</div>',unsafe_allow_html=True)


    #st.info("📌 Tip: Use the sidebar to switch between different plots. For deployment, push this to GitHub and host on Streamlit Cloud or Vercel.")
