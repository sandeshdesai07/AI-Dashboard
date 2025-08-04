import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
from math import sqrt

st.title("🧠 Model Training")

# Check if data is uploaded
if "df" in st.session_state:
    df = st.session_state.df

    # Target selection
    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    # Detect task type
    y_unique = df[target_column].nunique()
    if y_unique <= 20 and df[target_column].dtype in [int, object, 'category']:
        task_type = "classification"
    else:
        task_type = "regression"

    st.write(f"✅ Detected task type: **{task_type}**")

    # Train model only if not already trained
    if "trained_model" not in st.session_state or st.button("🔁 Retrain Model"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical columns if necessary
        X = pd.get_dummies(X)

        # Align features in case of missing dummy columns
        if "trained_features" in st.session_state:
            X = X.reindex(columns=st.session_state.trained_features, fill_value=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == "classification":
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Store model and feature names
        st.session_state.trained_model = model
        st.session_state.target_column = target_column
        st.session_state.task_type = task_type
        st.session_state.trained_features = X.columns.tolist()

        st.success("🎉 Model trained successfully!")

        # Show metrics
        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            st.write(f"🔍 **Accuracy:** {acc:.2f}")
        else:
            rmse = sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            st.write(f"📉 **RMSE:** {rmse:.2f}")
            st.write(f"📈 **R² Score:** {r2:.2f}")
    else:
        st.info("✅ Model already trained. You can go to the prediction page.")

else:
    st.warning("📂 Please upload a dataset from the Home page.")
