import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from math import sqrt

st.title("🧠 Model Training")

# Check if dataset is loaded
if "df" in st.session_state:
    df = st.session_state.df.copy()
    all_cols = df.columns.tolist()

    # User selects target variable
    target_col = st.selectbox("🎯 Select the target variable", all_cols)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Auto detect task type
        is_classification = y.dtype == "object" or y.nunique() <= 10

        # One-hot encode if necessary
        X = pd.get_dummies(X)

        # Align y if categorical
        if is_classification:
            y = y.astype("category").cat.codes

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store column info in session for prediction
        st.session_state.model_columns = X.columns.tolist()
        st.session_state.is_classification = is_classification

        # Train model
        if st.button("🚀 Train Model"):
            if is_classification:
                model = RandomForestClassifier()
            else:
                model = RandomForestRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state.trained_model = model

            # Evaluation
            if is_classification:
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Model trained successfully with Accuracy: {acc:.2f}")
            else:
                rmse = sqrt(mean_squared_error(y_test, y_pred))
                st.success(f"✅ Model trained successfully with RMSE: {rmse:.2f}")

    # Prediction Interface
    if "trained_model" in st.session_state:
        st.markdown("---")
        st.subheader("🔮 Make a Prediction")

        # Get user input dynamically based on training features
        input_data = {}
        for col in st.session_state.model_columns:
            input_data[col] = st.text_input(f"{col}", value="0")

        if st.button("Predict"):
            try:
                # Convert inputs to float
                input_df = pd.DataFrame([input_data])
                input_df = input_df.astype(float)

                prediction = st.session_state.trained_model.predict(input_df)[0]

                if st.session_state.is_classification:
                    st.success(f"🎉 Predicted class: {int(prediction)}")
                else:
                    st.success(f"🎉 Predicted value: {prediction:.2f}")

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

else:
    st.warning("📂 Please upload a dataset from the Home page.")
