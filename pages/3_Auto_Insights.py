import streamlit as st
import openai

st.title("🤖 AI-Powered Insights")

def get_ai_summary(df):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        prompt = f"Provide a detailed insight summary in bullet points from the following DataFrame:\n{df.head(10).to_string()}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Summary not available: {e}"

if "df" in st.session_state:
    df = st.session_state.df
    with st.spinner("Generating insights using AI..."):
        summary = get_ai_summary(df)
        st.markdown(summary)
else:
    st.warning("📂 Please upload a dataset from the Home page.")

