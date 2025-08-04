import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Auto Plots")

# ✅ Retrieve df from session_state
if "shared_df" in st.session_state:
    df = st.session_state["shared_df"]

    st.subheader("📈 Auto-Generated Insights")

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    if len(numeric_cols) >= 2:
        st.markdown("### 🔗 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    for col in numeric_cols:
        st.markdown(f"### Histogram: {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    for col in categorical_cols:
        st.markdown(f"### Count Plot: {col}")
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        st.pyplot(fig)

else:
    st.warning("📂 Please upload a dataset from the Home page.")
