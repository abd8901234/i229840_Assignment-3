import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="StockPredictor Pro", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main { background-color: #e8f6f3; }
        .stButton button {
            background-color: #008080;
            color: white;
            font-weight: bold;
        }
        .css-1d391kg { background-color: #003366 !important; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📊 StockPredictor Pro - Predicting Returns with ML")

# Session state initialization
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False
if "features_ready" not in st.session_state:
    st.session_state["features_ready"] = False
if "split_done" not in st.session_state:
    st.session_state["split_done"] = False
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
if "model" not in st.session_state:
    st.session_state["model"] = None
if "X_train" not in st.session_state:
    st.session_state["X_train"] = None
if "X_test" not in st.session_state:
    st.session_state["X_test"] = None
if "y_train" not in st.session_state:
    st.session_state["y_train"] = None
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()

# Welcome section
st.markdown("### 🚀 Welcome to StockPredictor Pro!")
st.image("https://media.giphy.com/media/3oriO0OEd9QIDdllqo/giphy.gif", width=400)
st.markdown("Use the sidebar to begin by selecting your stock data.")

# Sidebar inputs
st.sidebar.header("📥 Data Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., NFLX, GOOG, META):", "NFLX")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2022-01-01"))

# Fetch Data
if st.sidebar.button("📡 Fetch Stock Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        st.success("✅ Data Loaded Successfully!")
        st.dataframe(data.tail())
        st.session_state["data"] = data
        st.session_state["data_loaded"] = True
        st.session_state["features_ready"] = False
        st.session_state["split_done"] = False
        st.session_state["model_trained"] = False
        st.session_state["model"] = None
    else:
        st.error("❌ Data fetch failed. Please verify your ticker and date range.")

# Feature Engineering
if st.button("⚙ Feature Engineering"):
    if st.session_state["data_loaded"]:
        data = st.session_state["data"].copy()
        adj_close = None
        for col in ["Adj Close", "AdjClose", "Close"]:
            if col in data.columns:
                adj_close = col
                break

        if adj_close:
            data["Return"] = data[adj_close].pct_change()
            data["Lag1"] = data["Return"].shift(1)
            data.dropna(inplace=True)
            st.success("✅ Feature columns created!")
            st.line_chart(data["Return"])
            st.session_state["data"] = data
            st.session_state["features_ready"] = True
            st.session_state["split_done"] = False
            st.session_state["model_trained"] = False
        else:
            st.error("⚠ Could not find 'Adj Close' column.")
    else:
        st.warning("⚠ Please fetch data first.")

# Preprocessing
if st.button("🧹 Preprocessing"):
    if st.session_state["data_loaded"]:
        data = st.session_state["data"].dropna()
        st.success("✅ Null values removed.")
        st.write(data.describe())
        st.session_state["data"] = data
    else:
        st.warning("⚠ Load data first.")

# Split Data
if st.button("🧪 Train/Test Split"):
    if st.session_state["features_ready"]:
        data = st.session_state["data"]
        X = data[["Lag1"]]
        y = data["Return"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success("✅ Data split into train and test sets.")
        fig = px.pie(values=[len(X_train), len(X_test)], names=["Train", "Test"], title="Train/Test Split")
        st.plotly_chart(fig)
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["split_done"] = True
        st.session_state["model_trained"] = False
    else:
        st.warning("⚠ Run feature engineering first.")

# Model Training
if st.button("🤖 Train Model"):
    if st.session_state["split_done"]:
        model = LinearRegression()
        model.fit(st.session_state["X_train"], st.session_state["y_train"])
        st.session_state["model"] = model
        st.session_state["model_trained"] = True
        st.success("✅ Model trained!")
    else:
        st.warning("⚠ Run train/test split first.")

# Evaluate Model
if st.button("📈 Evaluate Model"):
    if st.session_state["model_trained"]:
        y_pred = st.session_state["model"].predict(st.session_state["X_test"])
        mse = mean_squared_error(st.session_state["y_test"], y_pred)
        r2 = r2_score(st.session_state["y_test"], y_pred)
        st.metric("📉 Mean Squared Error", f"{mse:.6f}")
        st.metric("📈 R² Score", f"{r2:.2f}")

        results_df = pd.DataFrame({"Actual": st.session_state["y_test"].values, "Predicted": y_pred})
        st.line_chart(results_df)
    else:
        st.warning("⚠ Please train the model first.")

# Visualize Results
if st.button("🔍 Visualize Predictions"):
    if st.session_state["model_trained"]:
        results = pd.DataFrame({
            "Actual": st.session_state["y_test"].values,
            "Predicted": st.session_state["model"].predict(st.session_state["X_test"])
        })
        fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs Predicted Returns")
        st.plotly_chart(fig)
    else:
        st.warning("⚠ Please train the model first.")
