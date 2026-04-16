from evaluation import evaluate_model
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from streamlit_autorefresh import st_autorefresh
import os
from evaluation import evaluate_model  # Make sure evaluation.py exists

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Stock Prediction", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1100px; margin:auto; }
h1,h2,h3 { text-align:center; }
</style>
""", unsafe_allow_html=True)

# ---------------- USER DB ----------------
USER_FILE = "users.csv"

def load_users():
    if not os.path.exists(USER_FILE):
        pd.DataFrame(columns=["username","password"]).to_csv(USER_FILE,index=False)
    return pd.read_csv(USER_FILE)

def save_user(u,p):
    df = load_users()
    df = pd.concat([df,pd.DataFrame([[u,p]],columns=["username","password"])],ignore_index=True)
    df.to_csv(USER_FILE,index=False)

# ---------------- SESSION ----------------
if "login" not in st.session_state:
    st.session_state.login=False
if "logout_msg" not in st.session_state:
    st.session_state.logout_msg=""
if "refresh" not in st.session_state:
    st.session_state.refresh=15

if st.session_state.logout_msg:
    st.success(st.session_state.logout_msg)
    st.session_state.logout_msg = "👋 Logout successful! Thanks for exploring real-time AI stock insights. Come back anytime!"

# ---------------- LOGIN PAGE ----------------
if not st.session_state.login:
    if st.session_state.logout_msg:
        st.success(st.session_state.logout_msg)
        st.session_state.logout_msg=""
        
    col1,col2,col3 = st.columns([2,3,2])
    
    with col1:
        st.image("https://img.freepik.com/free-vector/stock-market-concept-illustration_114360-1686.jpg")
    
    with col2:
        st.markdown("<h1>🤖 AI STOCK PREDICTION</h1>",unsafe_allow_html=True)
        menu = st.radio("Select Option",["Login","Register"])
        users_df = load_users()
        
        if menu=="Login":
            u = st.text_input("Username")
            p = st.text_input("Password",type="password")
            if st.button("Login"):
                if not users_df[(users_df.username==u)&(users_df.password==p)].empty:
                    st.session_state.login=True
                    st.rerun()
                else:
                    st.error("Invalid credentials ❌")
        else:
            u = st.text_input("New Username")
            p = st.text_input("New Password",type="password")
            if st.button("Register"):
                if u in users_df.username.values:
                    st.warning("User already exists ⚠️")
                else:
                    save_user(u,p)
                    st.success("Account created 🎉")
    
    with col3:
        st.image("https://img.freepik.com/free-vector/financial-growth-concept_23-2148789333.jpg")

# ---------------- DASHBOARD ----------------
if st.session_state.login:
    
    st.sidebar.title("📊 AI Stock")
    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.session_state.logout_msg="Logged out successfully 👋"
        st.rerun()
    st.session_state.refresh = st.sidebar.slider("Refresh (sec)",5,60,15)
    
    st.markdown("## 📊 Trading Dashboard")
    st.caption("Real-time stock insights with AI predictions")
    
    ticker = st.text_input("Enter Stock Ticker","AAPL").upper()
    
    # ---------------- Auto-refresh only for stock data ----------------
    st_autorefresh(interval=st.session_state.refresh*1000, key="stock_refresh")
    
    data = yf.download(ticker, period="1y", progress=False)
    if data.empty or len(data)<2:
        st.error("Invalid stock ❌")
        st.stop()
    
    # Indicators
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    delta = data["Close"].diff()
    gain = (delta.where(delta>0,0)).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain/loss
    data["RSI"] = 100-(100/(1+rs))
    
    last = float(data["Close"].iloc[-1])
    min_price = round(float(data["Close"].min()),2)
    max_price = round(float(data["Close"].max()),2)
    avg_price = round(float(data["Close"].mean()),2)
    
    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📊 Overview","📈 Charts","🤖 AI Prediction"])
    
    # -------- Overview --------
    with tab1:
        st.markdown("### 📊 Market Overview")
        kpi_html = f"""
        <div style="display:flex;gap:15px;">
            <div style="background:#1abc9c;padding:20px;border-radius:10px;color:white;flex:1;text-align:center">
                <h4>Min</h4><h2>{min_price}</h2></div>
            <div style="background:#3498db;padding:20px;border-radius:10px;color:white;flex:1;text-align:center">
                <h4>Max</h4><h2>{max_price}</h2></div>
            <div style="background:#9b59b6;padding:20px;border-radius:10px;color:white;flex:1;text-align:center">
                <h4>Avg</h4><h2>{avg_price}</h2></div>
            <div style="background:#e67e22;padding:20px;border-radius:10px;color:white;flex:1;text-align:center">
                <h4>Close</h4><h2>{round(last,2)}</h2></div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)
        
        st.markdown("### 📢 Trading Signal")
        rsi_value = float(data["RSI"].iloc[-1])
        if rsi_value>70:
            st.error("🔴 SELL Signal")
        elif rsi_value<30:
            st.success("🟢 BUY Signal")
        else:
            st.warning("🟡 HOLD Signal")
    
    # -------- Charts --------
    with tab2:
        st.subheader("📈 Candlestick Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                     low=data["Low"], close=data["Close"]))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)
        
        st.subheader("🥧 Price Distribution")
        labels = ["Min","Max","Avg","Close"]
        values = [min_price,max_price,avg_price,last]
        pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        st.plotly_chart(pie,use_container_width=True)
    
    # -------- AI Prediction --------
    with tab3:
        st.subheader("🤖 AI Prediction")

        # Scale data
        close_data = data["Close"].values.reshape(-1,1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_data)

        # Prepare sequences
        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = int(len(X)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Cached model
        @st.cache_resource
        def train_model(X_train, y_train):
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            return model

        model = train_model(X_train, y_train)

        # Predictions
        pred = model.predict(X_test)
        pred = scaler.inverse_transform(pred)
        real = scaler.inverse_transform(y_test.reshape(-1,1))

        # Evaluation
        mae, rmse, accuracy = evaluate_model(real, pred)
        st.subheader("📊 Model Evaluation")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("Accuracy", f"{accuracy:.2f}%")

        # Plot
        fig2, ax = plt.subplots()
        ax.plot(real, label="Actual")
        ax.plot(pred, label="Predicted")
        ax.legend()
        st.pyplot(fig2)

        # 7-Day Forecast
        future_input = scaled[-60:].reshape(1,60,1)
        future_preds = []
        for i in range(7):
            p = model.predict(future_input)
            future_preds.append(p[0,0])
            future_input = np.concatenate((future_input[:,1:,:], p.reshape(1,1,1)), axis=1)
        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

        forecast_df = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(7)],
            "Predicted Price": future_preds.flatten()
        })
        st.subheader("📅 7-Day Forecast")
        st.table(forecast_df)
        st.download_button("⬇ Download Forecast", forecast_df.to_csv(index=False), "forecast.csv", "text/csv")