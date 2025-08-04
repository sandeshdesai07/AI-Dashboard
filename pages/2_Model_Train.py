import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
# from joblib import dump  # Optional: For saving model

st.title("🧠 Model Training")

# Check if data is available
if "shared_df" not in st.session_state:
    st.warning("📂 Please upload a dataset from the Home page.")
    st.stop()

df = st.session_state.shared_df

# Select target column
target_col = st.selectbox("🎯 Select Target Variable", df.columns)

# Drop rows with missing values in target
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# Determine task type
task_type = st.radio("🔍 Task Type", ["Regression", "Classification"])

# Filter usable models
if task_type == "Regression":
    model_choice = st.selectbox("🛠️ Choose Regression Model", ["Linear Regression", "Random Forest Regressor"])
else:
    model_choice = st.selectbox("🛠️ Choose Classification Model", ["Logistic Regression", "Random Forest Classifier"])

# Handle categorical columns if any
X = pd.get_dummies(X)

# Split data
test_size = st.slider("🧪 Test Set Size (%)", min_value=10, max_value=50, value=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Train the model
if st.button("🚀 Train Model"):
    if task_type == "Regression":
        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()
    else:
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Model trained successfully!")

    # Save model (optional)
    # dump(model, "trained_model.joblib")
    # st.info("Model saved as trained_model.joblib")

    st.subheader("📊 Evaluation Metrics")

    if task_type == "Regression":
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
        st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.4f}")

        # Plot true vs predicted
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    else:
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)
