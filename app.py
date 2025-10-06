import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import datetime
from neuralprophet import NeuralProphet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ======================================
# 🧱 إعداد واجهة Streamlit
# ======================================
st.set_page_config(page_title="BONK Forecast Ensemble — Futures", layout="wide")
st.title("📈 BONK Forecast Ensemble — Futures (1000BONK)")

st.sidebar.header("⚙️ Settings")

symbol = st.sidebar.text_input("Symbol (Binance futures)", "1000BONK/USDT:USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1h","4h","1d"], index=2)
limit = st.sidebar.number_input("Candles to fetch", min_value=100, max_value=5000, value=1500)
forecast_horizon = st.sidebar.number_input("Forecast horizon (candles ahead)", min_value=1, max_value=50, value=6)

# ======================================
# 🏦 اختيار البورصة
# ======================================
exchange_choice = st.sidebar.selectbox(
    "📊 Exchange Source (Data Provider)",
    ["Binance Futures", "Bybit", "OKX", "Bitget", "KuCoin"],
    index=1  # نبدأ بـ Bybit لأنها تعمل في مصر
)

# إنشاء كائن المنصة
if exchange_choice == "Binance Futures":
    ex = ccxt.binanceusdm()
    symbol_format = symbol
elif exchange_choice == "Bybit":
    ex = ccxt.bybit()
    symbol_format = symbol.replace(":USDT", "")
elif exchange_choice == "OKX":
    ex = ccxt.okx()
    symbol_format = symbol
elif exchange_choice == "Bitget":
    ex = ccxt.bitget()
    symbol_format = symbol.replace(":USDT", "")
elif exchange_choice == "KuCoin":
    ex = ccxt.kucoin()
    symbol_format = symbol.split(":")[0]
else:
    st.error("❌ Exchange not supported.")
    st.stop()

# ======================================
# 🕒 تحميل البيانات
# ======================================
@st.cache_data(ttl=600)
def fetch_ohlcv(symbol, timeframe, limit):
    df = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(df, columns=["timestamp","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

try:
    df = fetch_ohlcv(symbol_format, timeframe, limit)
except Exception as e:
    st.error(f"⚠️ Failed to fetch OHLCV for {symbol_format} on {exchange_choice}:\n{e}")
    st.stop()

# ======================================
# 🧠 إعداد البيانات للنماذج
# ======================================
df["y"] = df["close"]
df["ds"] = df["datetime"]

# ======================================
# 🔮 نماذج التنبؤ
# ======================================

def run_neural_prophet(df, horizon):
    m = NeuralProphet(n_changepoints=30, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    m.fit(df[["ds","y"]], freq="auto", progress="off")
    future = m.make_future_dataframe(df, periods=horizon)
    forecast = m.predict(future)
    return forecast.tail(horizon)["yhat1"].values

def run_lightgbm(df, horizon):
    X = np.arange(len(df)).reshape(-1,1)
    y = df["close"].values
    model = LGBMRegressor()
    model.fit(X, y)
    future_X = np.arange(len(df), len(df)+horizon).reshape(-1,1)
    preds = model.predict(future_X)
    return preds

def run_xgboost(df, horizon):
    X = np.arange(len(df)).reshape(-1,1)
    y = df["close"].values
    model = XGBRegressor()
    model.fit(X, y)
    future_X = np.arange(len(df), len(df)+horizon).reshape(-1,1)
    preds = model.predict(future_X)
    return preds

# ======================================
# 🧩 تشغيل النماذج
# ======================================
run_neural = st.sidebar.checkbox("Run NeuralProphet (if installed)", True)
run_xgb = st.sidebar.checkbox("Run XGBoost (if installed)", True)
run_lgbm = st.sidebar.checkbox("Run LightGBM (if installed)", True)

if st.button("🚀 Run pipeline"):
    st.info(f"Fetching data from **{exchange_choice}** for **{symbol_format}** ...")
    last_close = df["close"].iloc[-1]

    results = []
    if run_neural:
        results.append(run_neural_prophet(df, forecast_horizon))
    if run_xgb:
        results.append(run_xgboost(df, forecast_horizon))
    if run_lgbm:
        results.append(run_lightgbm(df, forecast_horizon))

    if not results:
        st.warning("Please select at least one model.")
        st.stop()

    ensemble_pred = np.mean(results, axis=0)

    df_pred = pd.DataFrame({
        "datetime": pd.date_range(df["datetime"].iloc[-1], periods=forecast_horizon+1, freq=timeframe),
        "forecast": np.append([last_close], ensemble_pred)
    })

    # ======================================
    # 📊 رسم النتيجة
    # ======================================
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Historical"
    ))
    fig.add_trace(go.Scatter(
        x=df_pred["datetime"], y=df_pred["forecast"], 
        mode="lines+markers", name="Forecast", line=dict(color="yellow", width=3)
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ======================================
    # 🎯 تحديد الأهداف
    # ======================================
    last_price = last_close
    mean_pred = ensemble_pred.mean()
    if mean_pred > last_price:
        direction = "📈 LONG"
        tp1 = mean_pred * 1.01
        tp2 = mean_pred * 1.02
        tp3 = mean_pred * 1.03
        sl = last_price * 0.98
    else:
        direction = "📉 SHORT"
        tp1 = mean_pred * 0.99
        tp2 = mean_pred * 0.98
        tp3 = mean_pred * 0.97
        sl = last_price * 1.02

    st.subheader("🎯 Trading Signal")
    st.write(f"**Direction:** {direction}")
    st.write(f"**Take Profit 1:** {tp1:.6f}")
    st.write(f"**Take Profit 2:** {tp2:.6f}")
    st.write(f"**Take Profit 3:** {tp3:.6f}")
    st.write(f"**Stop Loss:** {sl:.6f}")

else:
    st.info("Adjust settings in sidebar and click **Run pipeline** to start.")
