import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("🚦 Smart Traffic Insights")

# Upload CSV
uploaded_file = st.file_uploader("Upload Traffic Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Show columns
    st.write("Available columns:", df.columns.tolist())

    # Select target column
    if "traffic_volume" in df.columns:
        target = "traffic_volume"
    else:
        target = st.selectbox("Select target column", df.columns)

    # Features = all except target
    X = df.drop(columns=[target])
    y = df[target]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = y.fillna(0)

    if X.shape[1] == 0:
        st.error("❌ No numeric features available for training!")
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)

        st.subheader("✅ Model Results")
        st.write(f"R² Score: {model.score(X_test, y_test):.2f}")

        # Plot actual vs predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.5)
        ax.set_xlabel("Actual Traffic Volume")
        ax.set_ylabel("Predicted Traffic Volume")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Predict on custom input
        st.subheader("🔮 Try Prediction")
        sample_input = {}
        for col in X.columns[:5]:  # take first 5 features for simplicity
            val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
            sample_input[col] = val

        if st.button("Predict Traffic"):
            input_df = pd.DataFrame([sample_input])
            st.write("Predicted Traffic Volume:", int(model.predict(input_df)[0]))
