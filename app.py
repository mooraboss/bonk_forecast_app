# app.py
# Streamlit app: BONK (futures) forecasting ensemble
#
# Features:
# - Fetch OHLCV from Binance (spot or futures)
# - EMA, RSI, ATR, VWAP
# - Forecast: Holt-Winters, ARIMA, NeuralProphet (optional), LightGBM/XGBoost (optional)
# - Ensemble (mean) and per-model signals (entry, SL, TP1/2/3)
# - Liquidity zones (volume clusters), Orderbook Imbalance (OBI)
# - All plots in unified window + separate NP plot if available
# - Graceful handling when optional libs are missing
#
# Usage: run `streamlit run app.py`

import streamlit as st
st.set_page_config(layout="wide", page_title="BONK Forecast Ensemble", initial_sidebar_state="expanded")

import math, time, warnings, traceback
from pprint import pprint
warnings.filterwarnings("ignore")

# --- imports that must exist for core functionality ---
try:
    import pandas as pd
    import numpy as np
    import ccxt
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception as e:
    st.error("Missing core packages. Ensure pandas, numpy, ccxt, matplotlib, statsmodels are installed.")
    st.exception(e)
    st.stop()

# --- optional libraries ---
HAS_NEURALPROPHET = False
HAS_XGBOOST = False
HAS_LIGHTGBM = False
try:
    from neuralprophet import NeuralProphet
    HAS_NEURALPROPHET = True
except Exception:
    HAS_NEURALPROPHET = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

# ---------- Helpers ----------
def make_exchange_futures():
    """
    Create ccxt exchange object configured for Binance Futures (USDM) if available,
    otherwise fallback to spot Binance (data will be spot).
    """
    # try binanceusdm or binance with defaultType=future
    for cls in ("binanceusdm","binance"):
        try:
            ex_cls = getattr(ccxt, cls)
            ex = ex_cls({"enableRateLimit": True})
            # prefer futures on regular binance by setting option
            if cls == "binance":
                ex.options = ex.options if hasattr(ex,'options') else {}
                ex.options['defaultType'] = 'future'
            return ex
        except Exception:
            continue
    # fallback
    return ccxt.binance({"enableRateLimit": True})

def fetch_ohlcv_futures(exchange, symbol, timeframe="15m", limit=1500):
    """
    Fetch OHLCV for futures or spot. Return DataFrame indexed by timezone-aware datetime (UTC).
    symbol: expect something like "1000BONK/USDT:USDT" or "BONK/USDT"
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        # if fails, try symbol without suffix or alternate naming
        alt = symbol.replace(":USDT","")
        try:
            ohlcv = exchange.fetch_ohlcv(alt, timeframe=timeframe, limit=limit)
        except Exception as e2:
            raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {e} / {e2}")
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("ds")
    df = df[["open","high","low","close","volume"]].astype(float)
    return df

# Indicators
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100/(1+rs))

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def compute_vwap(df):
    tp = (df['high'] + df['low'] + df['close'])/3
    return (tp * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-12)

# Liquidity zones (simple volume-price clustering)
def compute_liquidity_zones(df, window=300, bins=60, top_n=3, extend_pct=0.002):
    d = df.tail(window).copy()
    price = (d['high'] + d['low'])/2
    vol = d['volume']
    # histogram bins
    minp, maxp = price.min(), price.max()
    if maxp <= minp:
        return {"zones":[]}
    bins_edges = np.linspace(minp, maxp, bins+1)
    vol_per_bin = np.zeros(bins)
    for i in range(bins):
        mask = (price >= bins_edges[i]) & (price < bins_edges[i+1])
        vol_per_bin[i] = vol[mask].sum()
    top_idx = np.argsort(vol_per_bin)[-top_n:][::-1]
    zones = []
    for idx in top_idx:
        low = bins_edges[idx]
        high = bins_edges[idx+1]
        center = 0.5*(low+high)
        zones.append({"low":low*(1-extend_pct), "high":high*(1+extend_pct), "center":center, "vol":vol_per_bin[idx]})
    return {"zones":zones, "last_high": float(d['high'].max()), "last_low": float(d['low'].min())}

def compute_orderbook_imbalance(exchange, symbol, limit=100):
    try:
        ob = exchange.fetch_order_book(symbol, limit=limit)
        bids = sum([b[1] for b in ob.get('bids',[])])
        asks = sum([a[1] for a in ob.get('asks',[])])
        if bids + asks == 0:
            return None, ob
        return (bids - asks) / (bids + asks), ob
    except Exception:
        return None, None

# Forecast wrappers returning pd.Series indexed 1..horizon (RangeIndex)
def forecast_holt(series, horizon=6):
    try:
        model = ExponentialSmoothing(series.dropna(), trend="add", seasonal=None)
        fit = model.fit()
        f = fit.forecast(horizon)
        return pd.Series(f.values, index=pd.RangeIndex(1, horizon+1))
    except Exception:
        raise

def forecast_arima(series, horizon=6):
    try:
        m = ARIMA(series.dropna(), order=(2,1,2))
        f = m.fit()
        fo = f.forecast(steps=horizon)
        return pd.Series(fo.values, index=pd.RangeIndex(1, horizon+1))
    except Exception:
        raise

def forecast_neuralprophet(df, horizon=6, np_params=None):
    if not HAS_NEURALPROPHET:
        raise RuntimeError("NeuralProphet not installed.")
    # prepare
    df_np = df[['close']].reset_index().rename(columns={'ds':'ds','close':'y'})
    df_np['ds'] = pd.to_datetime(df_np['ds'])
    # only pass ds,y
    df_np = df_np[['ds','y']]
    # build model
    np_params = np_params or {}
    m = NeuralProphet(**np_params)
    # fit
    m.fit(df_np, freq=df.index.inferred_freq or "15min", progress="bar")
    future = m.make_future_dataframe(df_np, periods=horizon)
    forecast = m.predict(future)
    # forecast last horizon steps -> yhat1
    s = forecast.tail(horizon).reset_index(drop=True)['yhat1']
    # return as RangeIndex 1..horizon
    return pd.Series(s.values, index=pd.RangeIndex(1, horizon+1))

def features_for_ml(df):
    # simple lag features for ML models
    d = df.copy()
    d['ret1'] = d['close'].pct_change().fillna(0)
    d['ema21'] = ema(d['close'], 21)
    d['ema50'] = ema(d['close'], 50)
    d['rsi14'] = rsi(d['close'], 14)
    d['atr14'] = atr(d, 14)
    d = d.dropna()
    return d

def forecast_xgb(series, horizon=6):
    if not HAS_XGBOOST:
        raise RuntimeError("XGBoost not installed.")
    # simple autoregressive with lag features
    df = features_for_ml(series.to_frame(name='close'))
    if len(df) < 50:
        raise RuntimeError("Not enough data for XGBoost training.")
    # train simple regressor to predict next close
    X = df[['ret1','ema21','ema50','rsi14','atr14']].values
    y = df['close'].values
    # use last portion for training
    train_X, train_y = X[:-horizon], y[:-horizon]
    model = xgb.XGBRegressor(n_estimators=100, verbosity=0)
    model.fit(train_X, train_y)
    preds = []
    last_row = df.iloc[-1].copy()
    for i in range(horizon):
        feat = np.array([last_row['ret1'], last_row['ema21'], last_row['ema50'], last_row['rsi14'], last_row['atr14']])
        p = model.predict(feat.reshape(1,-1))[0]
        preds.append(p)
        # shift last_row (naive)
        last_row['ret1'] = (p - last_row['close'])/ (last_row['close']+1e-12)
        last_row['close'] = p
        last_row['ema21'] = 0.9*last_row['ema21'] + 0.1*p
        last_row['ema50'] = 0.98*last_row['ema50'] + 0.02*p
    return pd.Series(preds, index=pd.RangeIndex(1, horizon+1))

def forecast_lightgbm(series, horizon=6):
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM not installed.")
    df = features_for_ml(series.to_frame(name='close'))
    if len(df) < 50:
        raise RuntimeError("Not enough data for LightGBM training.")
    X = df[['ret1','ema21','ema50','rsi14','atr14']].values
    y = df['close'].values
    train_X, train_y = X[:-horizon], y[:-horizon]
    model = lgb.LGBMRegressor(n_estimators=200)
    model.fit(train_X, train_y)
    preds = []
    last_row = df.iloc[-1].copy()
    for i in range(horizon):
        feat = np.array([last_row['ret1'], last_row['ema21'], last_row['ema50'], last_row['rsi14'], last_row['atr14']])
        p = model.predict(feat.reshape(1,-1))[0]
        preds.append(p)
        last_row['ret1'] = (p - last_row['close'])/(last_row['close']+1e-12)
        last_row['close'] = p
        last_row['ema21'] = 0.9*last_row['ema21'] + 0.1*p
        last_row['ema50'] = 0.98*last_row['ema50'] + 0.02*p
    return pd.Series(preds, index=pd.RangeIndex(1, horizon+1))

# signal builder
def signal_from_forecast(preds, last_close, atr):
    mean_pred = float(preds.mean())
    direction = "LONG" if mean_pred > last_close else "SHORT"
    entry = float(last_close)
    if direction == "LONG":
        stop_loss = entry - 1.5 * float(atr)
        tp1 = entry + 0.5*(mean_pred - entry)
        tp2 = entry + 1.0*(mean_pred - entry)
        tp3 = mean_pred
    else:
        stop_loss = entry + 1.5 * float(atr)
        tp1 = entry - 0.5*(entry - mean_pred)
        tp2 = entry - 1.0*(entry - mean_pred)
        tp3 = mean_pred
    denom = abs(entry - stop_loss)
    if denom == 0:
        denom = 1e-12
    rr1 = abs((tp1-entry)/denom)
    rr2 = abs((tp2-entry)/denom)
    rr3 = abs((tp3-entry)/denom)
    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "atr": float(atr),
        "rr_tp1": rr1,
        "rr_tp2": rr2,
        "rr_tp3": rr3,
        "forecast_mean": mean_pred,
        "forecast_series": preds
    }

# ---------- Streamlit UI ----------
st.title("BONK Forecast Ensemble — Futures (1000BONK)")

with st.sidebar:
    st.markdown("## Settings")
    default_symbol = "1000BONK/USDT:USDT"
    symbol = st.text_input("Symbol (Binance futures)", value=default_symbol, help="Examples: 1000BONK/USDT:USDT or BONK/USDT")
    timeframe = st.selectbox("Timeframe", ["15m","5m","1h","1d"], index=0)
    limit = st.number_input("Candles to fetch", min_value=200, max_value=3000, value=1500, step=100)
    horizon = st.number_input("Forecast horizon (candles ahead)", min_value=1, max_value=24, value=6)
    run_np = st.checkbox("Run NeuralProphet (if installed)", value=HAS_NEURALPROPHET)
    run_xgb = st.checkbox("Run XGBoost (if installed)", value=HAS_XGBOOST)
    run_lgb = st.checkbox("Run LightGBM (if installed)", value=HAS_LIGHTGBM)
    preserve_plots = st.checkbox("Preserve all plots & outputs (true)", value=True)
    run_button = st.button("Run pipeline")

if not run_button:
    st.info("Enter parameters and press **Run pipeline**")
    st.stop()

# Execute pipeline
status = st.empty()
progress = st.progress(0)

try:
    status.text("Creating exchange...")
    ex = make_exchange_futures()
    progress.progress(5)

    status.text(f"Fetching {limit} candles for {symbol} @ {timeframe} ...")
    df = fetch_ohlcv_futures(ex, symbol, timeframe=timeframe, limit=limit)
    progress.progress(15)

    # compute indicators
    status.text("Computing indicators...")
    df['ema_21'] = ema(df['close'], 21)
    df['ema_50'] = ema(df['close'], 50)
    df['rsi_14'] = rsi(df['close'], 14)
    df['atr_14'] = atr(df, 14)
    df['VWAP'] = compute_vwap(df)
    progress.progress(30)

    # orderbook/OBI
    status.text("Fetching orderbook & computing OBI...")
    obi, orderbook = compute_orderbook_imbalance(ex, symbol, limit=200)
    progress.progress(40)

    # liquidity zones
    zones_out = compute_liquidity_zones(df, window=min(1200, len(df)), bins=80, top_n=3)
    progress.progress(45)

    # print last 5 candles with indicators
    status.text("Displaying last candles...")
    st.subheader("Last 5 candles with indicators")
    st.dataframe(df[['open','high','low','close','volume','ema_21','rsi_14','atr_14']].tail(5))
    progress.progress(50)

    # Forecasts per model (store pd.Series indexed 1..horizon)
    methods = {}
    errors = {}
    last_close = float(df['close'].iloc[-1])
    last_atr = float(df['atr_14'].iloc[-1])

    # Holt-Winters
    status.text("Holt-Winters forecasting...")
    try:
        methods['HoltWinters'] = forecast_holt(df['close'], horizon=horizon)
        status.text("Holt-Winters done.")
    except Exception as e:
        errors['HoltWinters'] = str(e)
        methods['HoltWinters'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))

    progress.progress(60)

    # ARIMA
    status.text("ARIMA forecasting...")
    try:
        methods['ARIMA'] = forecast_arima(df['close'], horizon=horizon)
        status.text("ARIMA done.")
    except Exception as e:
        errors['ARIMA'] = str(e)
        methods['ARIMA'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))

    progress.progress(70)

    # XGBoost
    if run_xgb and HAS_XGBOOST:
        status.text("XGBoost forecasting...")
        try:
            methods['XGBoost'] = forecast_xgb(df['close'], horizon=horizon)
        except Exception as e:
            errors['XGBoost'] = str(e)
            methods['XGBoost'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))
    else:
        methods['XGBoost'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))

    progress.progress(75)

    # LightGBM
    if run_lgb and HAS_LIGHTGBM:
        status.text("LightGBM forecasting...")
        try:
            methods['LightGBM'] = forecast_lightgbm(df['close'], horizon=horizon)
        except Exception as e:
            errors['LightGBM'] = str(e)
            methods['LightGBM'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))
    else:
        methods['LightGBM'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))

    progress.progress(80)

    # NeuralProphet
    if run_np and HAS_NEURALPROPHET:
        status.text("NeuralProphet preparing and fitting model (this can take time)...")
        try:
            # safe parameters
            np_params = dict(n_changepoints=30, yearly_seasonality=False, weekly_seasonality=False,
                             daily_seasonality=True, epochs=40, batch_size=64, learning_rate=1e-3)
            methods['NeuralProphet'] = forecast_neuralprophet(df, horizon=horizon, np_params=np_params)
            status.text("NeuralProphet done.")
        except Exception as e:
            errors['NeuralProphet'] = repr(e)
            methods['NeuralProphet'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))
    else:
        methods['NeuralProphet'] = pd.Series([np.nan]*horizon, index=pd.RangeIndex(1,horizon+1))

    progress.progress(90)

    # Ensemble: mean across available forecasts (ignore all-nan series)
    status.text("Computing ensemble...")
    valid = [s for s in methods.values() if s.notna().any()]
    if len(valid) == 0:
        raise RuntimeError("No forecasts available from any model.")
    # align: all are RangeIndex(1..h), safe to concat
    df_concat = pd.concat(valid, axis=1)
    ensemble = df_concat.mean(axis=1)
    progress.progress(95)

    # Build per-model signals
    per_model_signals = {}
    for name, preds in methods.items():
        per_model_signals[name] = signal_from_forecast(preds.fillna(method='ffill').fillna(last_close), last_close, last_atr)
        # include forecast series attached
        per_model_signals[name]['forecast_series'] = preds

    final_signal = signal_from_forecast(ensemble.fillna(last_close), last_close, last_atr)
    final_signal['forecast_series'] = ensemble

    progress.progress(98)
    status.text("Done. Preparing outputs...")

    # ---------- OUTPUT DISPLAY ----------
    st.subheader("Per-model Signals (Raw, detailed)")
    for name, sig in per_model_signals.items():
        st.markdown(f"**{name}**")
        cols = st.columns(2)
        with cols[0]:
            st.write({
                "direction": sig['direction'],
                "entry": sig['entry'],
                "stop_loss": sig['stop_loss'],
                "atr": sig['atr'],
                "forecast_mean": sig['forecast_mean']
            })
        with cols[1]:
            st.write({
                "tp1": sig['tp1'],
                "tp2": sig['tp2'],
                "tp3": sig['tp3'],
                "rr_tp1": sig['rr_tp1'],
                "rr_tp2": sig['rr_tp2'],
                "rr_tp3": sig['rr_tp3']
            })

    st.subheader("FINAL Ensemble Signal (Raw)")
    st.json({
        "direction": final_signal['direction'],
        "entry": final_signal['entry'],
        "stop_loss": final_signal['stop_loss'],
        "tp1": final_signal['tp1'],
        "tp2": final_signal['tp2'],
        "tp3": final_signal['tp3'],
        "atr": final_signal['atr'],
        "ensemble_mean": final_signal['forecast_mean']
    })

    st.subheader("Liquidity zones (top)")
    for i,z in enumerate(zones_out.get('zones',[])):
        st.write(f"Zone #{i+1}: center={z['center']:.8f}  vol={z['vol']:.0f}  [{z['low']:.8f} - {z['high']:.8f}]")

    st.write("Orderbook Imbalance (OBI):", obi)
    if orderbook:
        st.write("Orderbook top bids/asks (sample):")
        st.write({"bids": orderbook.get('bids',[])[:6], "asks": orderbook.get('asks',[])[:6]})

    # Save CSV (signals + metadata)
    outrow = final_signal.copy()
    outrow.update({
        "symbol": symbol,
        "time": str(df.index[-1]),
        "obi": obi
    })
    # add models means
    for name, sig in per_model_signals.items():
        outrow[f"{name}_mean"] = sig['forecast_mean']
        outrow[f"{name}_signal"] = sig['direction']
    dfout = pd.DataFrame([outrow])
    csv_name = f"bonk_futures_signal_{int(time.time())}.csv"
    dfout.to_csv(csv_name, index=False)
    st.success(f"Saved signal CSV: {csv_name}")
    st.download_button("Download CSV", dfout.to_csv(index=False), file_name=csv_name, mime="text/csv")

    # ---------- PLOTTING ----------
    st.subheader("Unified Forecast View (All Models + Ensemble + Liquidity zones)")
    fig, ax = plt.subplots(2,1, figsize=(12,9), gridspec_kw={"height_ratios":[3,1]})
    ax_price = ax[0]; ax_fc = ax[1]

    # Price
    ax_price.plot(df.index, df['close'], label="Close", color='black', linewidth=1)
    if 'VWAP' in df.columns:
        ax_price.plot(df.index, df['VWAP'], label="VWAP", color='orange', linewidth=1)

    # Forecast lines (plot short lines starting after last timestamp)
    last_ts = df.index[-1]
    # delta: infer frequency or use timeframe seconds
    try:
        if df.index.freq is not None:
            delta = df.index.freq
        else:
            # compute median diff
            delta = pd.to_timedelta(np.median(np.diff(df.index.values)).astype('timedelta64[ns]'))
    except Exception:
        delta = pd.Timedelta(minutes=15)

    # Plot each model forecast as a small line after last timestamp
    colors = {"HoltWinters":"cyan","ARIMA":"orange","XGBoost":"green","LightGBM":"purple","NeuralProphet":"magenta","Ensemble":"red"}
    for name, sig in per_model_signals.items():
        fs = sig['forecast_series']
        if fs is None or fs.isna().all():
            continue
        fc_index = [last_ts + (i+1)*delta for i in range(len(fs))]
        ax_price.plot(fc_index, fs.values, '--', color=colors.get(name,"gray"), label=f"{name} Forecast")
        # faint TP/SL lines
        ax_price.axhline(sig['stop_loss'], color=colors.get(name,'gray'), linestyle=':', alpha=0.15)
        ax_price.axhline(sig['tp1'], color=colors.get(name,'gray'), linestyle='--', alpha=0.2)

    # Ensemble line
    ensemble_series = final_signal['forecast_series']
    fc_index = [last_ts + (i+1)*delta for i in range(len(ensemble_series))]
    ax_price.plot(fc_index, ensemble_series.values, '-', color='red', linewidth=2, label='Ensemble (mean)')

    # Entry/SL/TPs for final signal
    ax_price.axhline(final_signal['entry'], color='purple', linestyle='-.', label=f"Entry {final_signal['entry']:.12f}")
    ax_price.axhline(final_signal['stop_loss'], color='red', linestyle=':', label=f"SL {final_signal['stop_loss']:.12f}")
    ax_price.axhline(final_signal['tp1'], color='green', linestyle='--', label=f"TP1 {final_signal['tp1']:.12f}")
    ax_price.axhline(final_signal['tp2'], color='green', linestyle='--', alpha=0.7, label=f"TP2 {final_signal['tp2']:.12f}")
    ax_price.axhline(final_signal['tp3'], color='green', linestyle='--', alpha=0.4, label=f"TP3 {final_signal['tp3']:.12f}")

    # Liquidity zones as horizontal spans
    N = min(300, len(df))
    for i,z in enumerate(zones_out.get('zones',[])):
        c = plt.cm.tab10(i)
        ax_price.axhspan(z['low'], z['high'], alpha=0.12, color=c, label=f"Liquidity zone #{i+1} (vol={z['vol']:.0f})")
        ax_price.hlines(z['center'], df.index[-N], df.index[-1], linestyles='dotted', colors=c, linewidth=1)

    ax_price.set_title(f"{symbol} Price & Forecasts")
    ax_price.legend(loc='upper left', fontsize=8)
    ax_price.grid(True)

    # ensemble steps plot
    ax_fc.plot(range(1, len(ensemble_series)+1), ensemble_series.values, marker='o', linestyle='-')
    ax_fc.set_xlabel("Forecast step (candles ahead)")
    ax_fc.set_title("Ensemble forecast steps")
    ax_fc.grid(True)

    st.pyplot(fig)

    # Separate NeuralProphet plot if available
    if HAS_NEURALPROPHET and run_np and per_model_signals.get('NeuralProphet', {}).get('forecast_series') is not None:
        np_fore = per_model_signals['NeuralProphet']['forecast_series']
        if not np_fore.isna().all():
            st.subheader("NeuralProphet Forecast (separate)")
            fc_index = [last_ts + (i+1)*delta for i in range(len(np_fore))]
            fig2, ax2 = plt.subplots(figsize=(12,5))
            ax2.plot(df.index, df['close'], label='Close', color='blue')
            ax2.plot(fc_index, np_fore.values, '--', label='NeuralProphet Forecast', color='magenta')
            ax2.axhline(per_model_signals['NeuralProphet']['stop_loss'], color='red', linestyle=':', label='NP SL')
            ax2.axhline(per_model_signals['NeuralProphet']['tp1'], color='green', linestyle='--', label='NP TP1')
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

    # Summary table
    st.subheader("Summary of all model signals")
    rows = []
    for name, sig in per_model_signals.items():
        rows.append({
            "Model": name,
            "Signal": sig['direction'],
            "Entry": sig['entry'],
            "SL": sig['stop_loss'],
            "TP1": sig['tp1'],
            "TP2": sig['tp2'],
            "TP3": sig['tp3'],
            "MeanForecast": sig['forecast_mean']
        })
    rows.append({
        "Model": "Ensemble",
        "Signal": final_signal['direction'],
        "Entry": final_signal['entry'],
        "SL": final_signal['stop_loss'],
        "TP1": final_signal['tp1'],
        "TP2": final_signal['tp2'],
        "TP3": final_signal['tp3'],
        "MeanForecast": final_signal['forecast_mean']
    })
    st.dataframe(pd.DataFrame(rows).set_index("Model"))

    # errors if any
    if errors:
        st.warning("Some models failed. See details:")
        st.json(errors)

    progress.progress(100)
    status.text("Pipeline finished: all outputs produced (no rounding applied).")

except Exception as e:
    status.error("Pipeline failed:")
    st.exception(e)
    st.stop()
