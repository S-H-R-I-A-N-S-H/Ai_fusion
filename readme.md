# 🚦 Smart Traffic Insights

## Overview

This project provides a **Streamlit web app** for predicting and analyzing traffic volume using machine learning. The app allows users to:

* Upload a traffic dataset (CSV format)
* Train a **Linear Regression model** on the provided data
* View model performance (R² score)
* Compare **actual vs predicted traffic volumes** with visualizations
* Perform **custom predictions** by inputting feature values interactively

The app is designed to be simple, interactive, and easy to use, making it a great starting point for traffic forecasting and congestion analysis.

---

## Features

* 📊 Dataset preview and column detection
* 🔧 Automatic feature/target separation (numeric features only)
* 🧹 Data cleaning: fills missing values with 0
* ✨ Train/Test split (80/20)
* 🤖 Machine Learning: **Linear Regression** model
* 📈 Visualization: Actual vs Predicted scatter plot
* 🔮 Interactive prediction with custom input values

---

## Installation

Clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

* Streamlit
* Scikit-learn
* Pandas
* Numpy
* Matplotlib

(See `requirements.txt` for details)

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the provided URL in your browser.

---

## Input Data Format

The uploaded CSV should contain:

* **Target column:** `traffic_volume` (or you can manually select another column)
* **Feature columns:** Any numeric features (e.g., timestamp-derived features, junction IDs, weather data, etc.)

Example structure:

| date\_time          | junction | traffic\_volume | temperature | humidity |
| ------------------- | -------- | --------------- | ----------- | -------- |
| 2025-01-01 08:00:00 | 1        | 320             | 22.5        | 60       |
| 2025-01-01 09:00:00 | 1        | 450             | 23.0        | 58       |

---

## Model

* **Algorithm:** Linear Regression (from scikit-learn)
* **Evaluation Metric:** R² score (explained variance)
* **Training Strategy:** 80% training, 20% testing split

---

## Future Enhancements

* Feature engineering: extract time-based features (hour, day, weekday, etc.)
* Outlier detection & normalization
* Support for multiple ML models (Random Forest, XGBoost, Neural Nets)
* Predict congestion levels (e.g., scale of 1–5)
* Identify peak traffic hours

---

## Author

Built for smart traffic analytics 🚦 using **Python, Streamlit, and Scikit-learn**.
