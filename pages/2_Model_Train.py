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

    # Evaluation
    st.subheader("📊 Evaluation Metrics")

    if task_type == "Regression":
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
        st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.4f}")

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

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

    # Prediction Input
    st.subheader("🔍 Make a Prediction")

    input_data = {}
    for col in X.columns:
        dtype = df[col].dtype
        if dtype == 'object':
            val = st.selectbox(f"{col}:", df[col].unique())
        else:
            val = st.number_input(f"{col}:", value=float(X[col].mean()))
        input_data[col] = val

    input_df = pd.DataFrame([input_data])

    # Align columns with training data (in case of dummies)
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    if st.button("🔮 Predict on New Data"):
        prediction = model.predict(input_df)[0]
        st.success(f"🧾 Predicted {target_col}: {prediction}")
