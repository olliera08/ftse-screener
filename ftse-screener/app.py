import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ---------- Page setup ----------
st.set_page_config(page_title="FTSE 100 Screener", layout="wide")
st.title("FTSE 100 Screener")
st.caption("Educational project. Not financial advice.")


# ---------- Step 1: Get the data ----------
@st.cache_data(ttl=3600)  # remembers the data for 1 hour so it doesn't re-download on every click
def load_data():
    # Read our list of companies
    companies = pd.read_csv("ftse100.csv")
    tickers = companies["ticker"].tolist()

    # Get company info (P/E, ROE, dividend) one company at a time
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}  # if it fails, carry on with empty info
        rows.append({
            "ticker": t,
            "pe": info.get("trailingPE"),
            "roe": info.get("returnOnEquity"),
            "div_yield": info.get("dividendYield"),
        })
    fundamentals = pd.DataFrame(rows)

    # Get 2 years of daily share prices
    prices = yf.download(tickers, period="2y", auto_adjust=True, progress=False)["Close"]

    # Work out 12-month return and volatility
    daily_changes = prices.pct_change()
    metrics = pd.DataFrame(index=prices.columns)
    metrics["ret_12m"] = prices.iloc[-1] / prices.iloc[-252] - 1
    metrics["vol_1y"] = daily_changes.tail(252).std() * np.sqrt(252)
    metrics.index.name = "ticker"
    metrics = metrics.reset_index()

    # Join everything into one table
    table = companies.merge(fundamentals, on="ticker").merge(metrics, on="ticker")
    return table

@st.cache_data(ttl=3600)
def load_history(ticker):
    return yf.Ticker(ticker).history(period="2y")["Close"]


# ---------- Step 2: Score the companies ----------
def score_companies(table, weights):
    # Turn each column into a rank from 0 to 1
    ranks = table[list(weights)].rank(pct=True)

    # For these, a LOWER number is better, so flip them
    for col in ["pe", "vol_1y"]:
        if col in ranks:
            ranks[col] = 1 - ranks[col]

    # Missing data gets a neutral score of 0.5
    ranks = ranks.fillna(0.5)

    # Weighted average, scaled to 0-100
    total = 0
    for col, weight in weights.items():
        total = total + ranks[col] * weight
    total = total / sum(weights.values())
    return (total * 100).round(1)


# ---------- Step 3: Build the page ----------
with st.spinner("Downloading market data..."):
    table = load_data()
    
st.sidebar.header("Filters")
sectors = st.sidebar.multiselect("Sectors (leave empty for all)", sorted(table["sector"].unique()))
if sectors:
    table = table[table["sector"].isin(sectors)] 

st.sidebar.header("How much does each factor matter?")
weights = {
    "div_yield": st.sidebar.slider("Income (dividend yield)", 0, 10, 0),
    "pe": st.sidebar.slider("Value (low P/E)", 0, 10, 5),
    "roe": st.sidebar.slider("Quality (high ROE)", 0, 10, 5),
    "ret_12m": st.sidebar.slider("Momentum (12m return)", 0, 10, 3),
    "vol_1y": st.sidebar.slider("Low volatility", 0, 10, 2),
}
weights = { name: w for name, w in weights.items() if w > 0}  # ignore sliders set to 0

if weights:
    table["score"] = score_companies(table, weights)
    table = table.sort_values("score", ascending=False)
    columns = ["name", "sector", "score", "pe", "roe", "ret_12m", "vol_1y"]
else:
    st.warning("Move at least one slider above 0.")
    columns = ["name", "sector", "pe", "roe", "ret_12m", "vol_1y"]

st.subheader("Ranked companies")
st.dataframe(
    table[columns].style.format(
        {"pe": "{:.1f}", "roe": "{:.1%}", "ret_12m": "{:.1%}", "vol_1y": "{:.1%}"},
        na_rep="-",
    ),
    use_container_width=True,)

st.download_button(
"Download results as CSV",
table[columns].to_csv(index=False),
file_name="ftse_screen.csv",
mime="text/csv",   
)

st.subheader("Risk vs return")
chart = px.scatter(
    table, x="vol_1y", y="ret_12m", color="sector",
    hover_name="name", labels={"vol_1y": "Volatility", "ret_12m": "12-month return"},
)
st.plotly_chart(chart, use_container_width=True)

st.subheader("Company deep dive")
choice = st.selectbox("Pick a company", table["name"])
ticker = table.loc[table["name"] == choice, "ticker"].iloc[0]
history = load_history(ticker)

chart_data = history.to_frame("Price")
chart_data["50-day average"] = history.rolling(50).mean()
chart_data["200-day average"] = history.rolling(200).mean()
st.line_chart(chart_data)