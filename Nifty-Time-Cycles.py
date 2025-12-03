import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go

# -------------------------------------------------------------
#                TIME CYCLE FUNCTIONS
# -------------------------------------------------------------

def get_nifty_daily(start="2010-01-01", end=None):
    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    df = yf.download("^NSEI", start=start, end=end, progress=False)

    if df.empty:
        st.error("❌ Failed to download NIFTY data.")
        return pd.DataFrame()

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index = pd.to_datetime(df.index)
    return df


def find_pivots(df, lookback=3):
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_high[i] = True
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def get_last_pivot_index(df, pivot_type="low"):
    col = "swing_low" if pivot_type == "low" else "swing_high"
    pivots = df.index[df[col]].tolist()
    if not pivots:
        return None
    last_pivot_date = pivots[-1]
    return df.index.get_loc(last_pivot_date)


def project_time_cycles(df, pivot_idx, cycles):
    pivot_row = df.iloc[pivot_idx]
    pivot_date = df.index[pivot_idx]
    pivot_price = pivot_row["close"]

    rows = []
    for c in cycles:
        idx = pivot_idx + c
        if idx >= len(df):
            continue

        row = df.iloc[idx]
        close = row["close"]

        abs_change = close - pivot_price
        pct_change = (abs_change / pivot_price) * 100 if pivot_price != 0 else 0
        body = close - row["open"]
        candle = "Bullish" if body > 0 else "Bearish" if body < 0 else "Doji"

        rows.append(
            {
                "Cycle Bars": c,
                "Cycle Date": df.index[idx].date(),
                "Pivot Price": round(float(pivot_price), 2),
                "Price @ Cycle": round(float(close), 2),
                "Abs Change": round(float(abs_change), 2),
                "% Change": round(float(pct_change), 2),
                "Candle": candle,
            }
        )

    return pd.DataFrame(rows)

# -------------------------------------------------------------
#                  STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="NIFTY Time Cycles (3-6-9 / 30-60-90)", layout="wide")
st.title("📈 NIFTY Daily Time Cycle Scanner")
st.markdown("### 3-6-9, 30-60-90, 300-600-900 Cycle Reversal Zones")

st.info("This tool automatically detects the latest pivot and projects reversal time cycles.")

# -------------------------------
# FETCH DATA
# -------------------------------
with st.spinner("Fetching NIFTY data..."):
    df = get_nifty_daily(start="2015-01-01")

if df.empty:
    st.stop()

with st.spinner("Detecting pivots..."):
    df = find_pivots(df)

# -------------------------------
# SELECT PIVOT TYPE
# -------------------------------
pivot_type = st.radio(
    "Select Pivot Type:",
    ["low", "high"],
    horizontal=True,
)

pivot_idx = get_last_pivot_index(df, pivot_type)

if pivot_idx is None:
    st.error(f"No {pivot_type} pivot found.")
    st.stop()

pivot_date = df.index[pivot_idx]
pivot_price = df.iloc[pivot_idx]["close"]

# SAFE FLOAT CONVERSION
try:
    pivot_price = float(pivot_price)
except:
    st.error("Pivot price is invalid.")
    st.stop()

st.success(
    f"Last pivot **{pivot_type.upper()}** at **{pivot_date.date()}** | "
    f"Price: **{pivot_price:.2f}**"
)

# -------------------------------
# CYCLE PROJECTION
# -------------------------------
cycles = [3, 6, 9, 30, 60, 90, 300, 600, 900]
cycle_df = project_time_cycles(df, pivot_idx, cycles)

# -------------------------------
# PLOT CHART
# -------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="NIFTY"
))

# Pivot line
fig.add_vline(
    x=pivot_date,
    line_width=2,
    line_dash="dash",
    line_color="blue"
)

# Cycle lines
for _, row in cycle_df.iterrows():
    fig.add_vline(
        x=row["Cycle Date"],
        line_width=1,
        line_dash="dot",
        line_color="red"
    )

fig.update_layout(
    title="NIFTY Time Cycles Projection",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TABLE OUTPUT
# -------------------------------
st.subheader("📊 Cycle Projection Table")
st.dataframe(cycle_df, use_container_width=True)

st.download_button(
    "Download Cycle Table (CSV)",
    data=cycle_df.to_csv(index=False),
    file_name="nifty_time_cycles.csv",
    mime="text/csv"
)
