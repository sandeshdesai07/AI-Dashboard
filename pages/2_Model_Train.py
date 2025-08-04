import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

st.title("🤖 Train ML Model")

# Check if dataset is available
if "df" not in st.session_state:
    st.warning("📂 Please upload a dataset from the Home page.")
    st.stop()

df = st.session_state.df.copy()

# Select target column
target_col = st.selectbox("🎯 Select target column", df.columns)

feature_cols = [col for col in df.columns if col != target_col]
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df[target_col]

# Infer task type
if y.dtype in ['int64', 'float64'] and y.nunique() > 10:
    task_type = "Regression"
else:
    task_type = "Classification"

st.markdown(f"🔍 Detected task: **{task_type}**")

# Choose model
if task_type == "Regression":
    model_name = st.radio("📦 Choose model", ["Linear Regression", "Random Forest Regressor"])
else:
    model_name = st.radio("📦 Choose model", ["Logistic Regression", "Random Forest Classifier"])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model on button click
if st.button("🚀 Train Model"):
    if task_type == "Regression":
        if model_name == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()
    else:
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Model trained successfully!")

    # Show metrics
    st.subheader("📊 Model Evaluation")
    if task_type == "Regression":
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.4f}")
        st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True))
    else:
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔁 Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

    # Save model to session state
    st.session_state.model = model
    st.session_state.features = X.columns.tolist()

    # Prediction
    st.subheader("🔍 Make a Prediction")
    input_data = []
    for col in st.session_state.features:
        val = st.text_input(f"Enter value for **{col}**")
        input_data.append(val)

    if st.button("📈 Predict"):
        try:
            input_array = np.array(input_data, dtype=np.float64).reshape(1, -1)
            pred = model.predict(input_array)
            st.success(f"📌 Prediction: {pred[0]}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
