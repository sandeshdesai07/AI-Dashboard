import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import io
import base64
import requests

################################PAGE TITLE###################################
st.set_page_config(page_title="AI Analytics Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .dashboard-title {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(90deg, #84CFDB, #ffffff);
        display: inline-block;
        padding: 10px 30px;
        border-radius: 15px;
        box-shadow: 2px 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease-in-out;
    }

    .dashboard-title:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #B0E0E6, #ffffff);
        box-shadow: 4px 6px 20px rgba(0, 0, 0, 0.15);
    }
    </style>

    <div style="text-align: center; padding: 10px;">
        <h1 class="dashboard-title">
            📈 AI-Powered Analytics Dashboard
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)


################################FILE UPLOAD SECTION#########################
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
        border-radius: 30px;
        background-color: #f9f9f9;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- File Upload ----------            

with st.container():
    st.markdown(
        """
        <style>
        .upload-title-container {
            display: inline-block;
            background: linear-gradient(135deg, #e0f7fa, #ffffff);
            padding: 8px 20px;
            border-radius: 50px;
            border: 5px solid #b2ebf2;
            box-shadow: 2px 4px 12px rgba(0, 0, 0, 0.05);
            transition: all 0.4s ease-in-out;
        }

        .upload-title-container:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, #b2ebf2, #e0f7fa);
            box-shadow: 4px 8px 18px rgba(0, 0, 0, 0.15);
            border-color: #4dd0e1;
        }

        .upload-title-text {
            color: #00796b;
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            transition: color 0.4s ease;
        }

        .upload-title-container:hover .upload-title-text {
            color: #004d40;
        }
        </style>

        <div style="text-align: center;">
            <div class="upload-title-container">
                <h3 class="upload-title-text">🪄 Upload and Understand Your Data</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
  
    st.markdown(
    "<h5 style='text-align: center; color: #AE6BC4;'>Upload your dataset and explore its structure and basic statistics.</h5>",
    unsafe_allow_html=True
)
    st.write("")
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    
############################Main Operation###################################

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

    #st.subheader("📈 Auto Plots")
    #e3f2fd
    st.markdown("""
    <div style="
        background: linear-gradient(to right,#ABF091, #bbdefb);
        padding: 10px 20px;
        border-left: 6px solid #2196f3;
        border-radius: 8px;
        font-family: 'Segoe UI', sans-serif;
        color: #0d47a1;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    ">
        📈 Auto Plots
    </div>
""", unsafe_allow_html=True)

    ############################Option Panel Hover##########################
    st.sidebar.markdown("""
    <style>
        .custom-sidebar-title {
            background: linear-gradient(to right, #98CEB8, #98A4CE);
            padding: 3px 7px;
            border-radius: 15px;
            font-family: 'Trebuchet MS', sans-serif;
            color: #5F2D2D;
            font-size: 22px;
            font-weight: bold;
            text-align: center;

            /* Neutral states for animatable props */
            box-shadow: 0 0 0px rgba(0, 0, 0, 0.0);
            transform: scale(1);
            text-shadow: none;

            /* Add transitions */
            transition: 
                background 1.2s ease,
                color 1.2s ease,
                box-shadow 1.2s ease,
                transform 1.2s ease,
                text-shadow 1.2s ease;
        }

        .custom-sidebar-title:hover {
            background: linear-gradient(to right, #B8E5D1, #C5B8E5);
            color: #2E1C1C;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.25);
            transform: scale(1.05);
            text-shadow: 0 0 5px #fff;
        }
    </style>

    <div class="custom-sidebar-title">
        🛠️ Options Panel
    </div>
""", unsafe_allow_html=True)

##########################################################
    
    #///////////////////////option panel//////////////////
    with st.container():
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        st.markdown("""
    <style>
        /* Target the selectbox label */
        .custom-selectbox-label {
            background: linear-gradient(to right, #AEE2FF, #C2FFD9);
            padding: 8px 12px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Trebuchet MS', sans-serif;
            color: #333;
            transition: all 1s ease;
            display: inline-block;
        }

        .custom-selectbox-label:hover {
            background: linear-gradient(to right, #FDDDE6, #D5F4E6);
            color: #1a1a1a;
            transform: scale(1.03);
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

# Render the styled label manually
st.sidebar.markdown('<div class="custom-selectbox-label">📊 Choose a Plot Type</div>', unsafe_allow_html=True)

# Display the selectbox without label
plot_type = st.sidebar.selectbox("", ["Correlation Heatmap", "Missing Values Heatmap", "Histogram", "Boxplot", "Countplot", "Pairplot", "Violinplot"])
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #st.write("")
     
    #plot_type = st.sidebar.selectbox("📊 Choose a Plot Type", ["Correlation Heatmap", "Missing Values Heatmap","Histogram", "Boxplot", "Countplot", "Pairplot", "Violinplot"])

    
    
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
