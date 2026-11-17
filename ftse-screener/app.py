import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

# ---------- Settings ----------
TRADING_DAYS = 252  # roughly how many days the market is open in a year

# Every factor you can score on: column name -> label shown on the slider
FACTORS = {
    "pe": "Value (low P/E)",
    "roe": "Quality (high ROE)",
    "profit_margin": "Profitability (profit margin)",
    "debt_to_equity": "Low debt (debt/equity)",
    "div_yield": "Income (dividend yield)",
    "mom_12_1": "Momentum (12m return, skipping last month)",
    "ret_3m": "Recent trend (3m return)",
    "vol_1y": "Low volatility",
    "max_dd": "Small drawdown (smaller worst fall)",
}

# For these, a LOWER number is better, so the ranking is flipped when scoring
LOWER_IS_BETTER = ["pe", "debt_to_equity", "vol_1y"]

DEFAULT_WEIGHTS = {"pe": 5, "roe": 5, "mom_12_1": 3, "vol_1y": 2}

# How each column is shown in the table
NUMBER_FORMATS = {
    "market_cap_bn": "{:.1f}",
    "score": "{:.1f}",
    "pe": "{:.1f}",
    "roe": "{:.1%}",
    "profit_margin": "{:.1%}",
    "debt_to_equity": "{:.0f}",
    "div_yield": "{:.1%}",
    "mom_12_1": "{:.1%}",
    "ret_3m": "{:.1%}",
    "ret_12m": "{:.1%}",
    "vol_1y": "{:.1%}",
    "max_dd": "{:.1%}",
}

st.set_page_config(page_title="FTSE 100 Screener", layout="wide")
st.title("FTSE 100 Screener")
st.caption("Educational project using Yahoo Finance data. Not financial advice.")



# ---------- Step 1: Get the data ----------
CACHE_TABLE = "cache_table.csv"
CACHE_PRICES = "cache_prices.csv"
CACHE_MAX_AGE_HOURS = 24


def fetch_one(ticker):
    """Get company info for one ticker. Retries if Yahoo gives a blank answer."""
    info = {}
    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        if len(info) > 5:  # a real answer has lots of fields
            break
        time.sleep(1 + attempt)  # wait a little longer each retry

    # Yahoo has changed how it reports dividend yield (0.035 vs 3.5), so we check
    div = info.get("dividendYield")
    if div is not None and div > 0.5:
        div = div / 100

    return {
        "ticker": ticker,
        "market_cap_bn": (info.get("marketCap") or np.nan) / 1e9,
        "pe": info.get("trailingPE"),
        "roe": info.get("returnOnEquity"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "div_yield": div,
    }


def calculate_metrics(prices):
    """Turn a table of daily prices into one row of metrics per company."""
    prices = prices.dropna(how="all").ffill()
    daily_changes = prices.pct_change()
    blank = pd.Series(np.nan, index=prices.columns)

    def change_over(days_ago, from_days_ago=0):
        # Price change between two points in time (blank if not enough history)
        if len(prices) <= days_ago:
            return blank
        end = prices.iloc[-1 - from_days_ago]
        start = prices.iloc[-1 - days_ago]
        return end / start - 1

    metrics = pd.DataFrame(index=prices.columns)
    metrics["ret_3m"] = change_over(63)
    metrics["ret_12m"] = change_over(TRADING_DAYS)
    # 12-1 momentum: the last month is skipped, which is common in finance research
    metrics["mom_12_1"] = change_over(TRADING_DAYS, from_days_ago=21)
    metrics["vol_1y"] = daily_changes.tail(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)
    # Max drawdown: the biggest fall from a peak to a later low
    metrics["max_dd"] = prices.tail(TRADING_DAYS).apply(lambda s: (s / s.cummax() - 1).min())
    metrics.index.name = "ticker"
    return metrics.reset_index()



def download_prices(tickers, period="5y", chunk_size=25):
    """Download prices in batches of 25 so one bad ticker can't ruin everything."""
    parts = []
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i + chunk_size]
        part = yf.download(batch, period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(part, pd.Series):  # happens if only one ticker comes back
            part = part.to_frame(batch[0])
        parts.append(part)
    return pd.concat(parts, axis=1)


def cache_is_fresh():
    """True if we saved data to disk in the last 24 hours."""
    if not (os.path.exists(CACHE_TABLE) and os.path.exists(CACHE_PRICES)):
        return False
    age_hours = (time.time() - os.path.getmtime(CACHE_TABLE)) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


@st.cache_data(ttl=3600)  # remembers the data for 1 hour while the app is open
def load_data():
    # If we downloaded recently, reuse the saved files (survives closing the app)
    if cache_is_fresh():
        table = pd.read_csv(CACHE_TABLE)
        prices = pd.read_csv(CACHE_PRICES, index_col=0, parse_dates=True)
        return table, prices

    companies = pd.read_csv("ftse100.csv")
    tickers = companies["ticker"].tolist()

    with ThreadPoolExecutor(max_workers=6) as pool:
        fundamentals = pd.DataFrame(list(pool.map(fetch_one, tickers)))

    prices = download_prices(tickers)
    metrics = calculate_metrics(prices)
    table = companies.merge(fundamentals, on="ticker").merge(metrics, on="ticker")

    # Only save if the download looks healthy (not a rate-limited failure)
    if table["pe"].notna().sum() > 20:
        table.to_csv(CACHE_TABLE, index=False)
        prices.to_csv(CACHE_PRICES)

    return table, prices


# ---------- Step 2: Score the companies ----------
def score_companies(table, weights, sector_relative):
    """Give every company a score from 0 to 100."""
    columns = list(weights)

    if sector_relative:
        # Rank each company only against others in its own sector
        ranks = table.groupby("sector")[columns].rank(pct=True)
        sector_size = table.groupby("sector")["ticker"].transform("size")
        ranks.loc[sector_size < 3, :] = np.nan  # too few peers to rank fairly
    else:
        ranks = table[columns].rank(pct=True)

    for col in columns:
        if col in LOWER_IS_BETTER:
            ranks[col] = 1 - ranks[col]

    ranks = ranks.fillna(0.5)  # missing data gets a neutral score

    total = sum(ranks[col] * weight for col, weight in weights.items())
    return (total / sum(weights.values()) * 100).round(1)


# ---------- Step 3: Load data ----------
with st.spinner("Downloading market data (this can take a minute the first time)..."):
    table, prices = load_data()

if table["pe"].isna().all():
    st.warning("Yahoo returned no company data. It may be rate-limiting you. Wait a few minutes and press Refresh.")

fundamental_cols = ["pe", "roe", "profit_margin", "debt_to_equity", "div_yield"]
no_company_data = table[table[fundamental_cols].isna().all(axis=1)]
no_prices = table[table["ret_12m"].isna()]

with st.expander(
    f"Data quality: {len(no_prices)} without prices, {len(no_company_data)} without company data"
):
    st.write("**No price history** (wrong ticker, or listed less than a year ago):")
    st.write(", ".join(no_prices["name"]) or "None")
    st.write("**No company data** (Yahoo failed, or an investment trust):")
    st.write(", ".join(no_company_data["name"]) or "None")
    st.caption(
        "If a normal company is listed here every time, check its ticker in ftse100.csv "
        "on finance.yahoo.com."
    )

# ---------- Step 4: Sidebar controls ----------
st.sidebar.header("Filters")
chosen_sectors = st.sidebar.multiselect("Sectors (empty = all)", sorted(table["sector"].unique()))
min_cap = st.sidebar.slider("Minimum market cap (£bn)", 0, 100, 0)
max_pe = st.sidebar.slider("Maximum P/E", 5, 100, 100)

st.sidebar.header("Factor weights")
st.sidebar.caption("0 = ignore this factor. Higher = matters more.")
weights = {}
for column, label in FACTORS.items():
    weight = st.sidebar.slider(label, 0, 10, DEFAULT_WEIGHTS.get(column, 0))
    if weight > 0:
        weights[column] = weight

sector_relative = st.sidebar.checkbox(
    "Compare within each sector",
    help="A bank and a drugmaker shouldn't be judged on P/E against each other. "
         "This ranks each company only against its own sector.",
)

if st.sidebar.button("Refresh data"):
    for saved_file in (CACHE_TABLE, CACHE_PRICES):
        if os.path.exists(saved_file):
            os.remove(saved_file)
    st.cache_data.clear()
    st.rerun()

# ---------- Step 5: Apply filters, then score ----------
filtered = table.copy()
if chosen_sectors:
    filtered = filtered[filtered["sector"].isin(chosen_sectors)]
filtered = filtered[filtered["market_cap_bn"].fillna(0) >= min_cap]
filtered = filtered[filtered["pe"].isna() | (filtered["pe"] <= max_pe)]  # keep missing P/E

columns = ["name", "sector", "market_cap_bn", "pe", "roe", "div_yield", "ret_12m", "vol_1y"]

if filtered.empty:
    st.warning("No companies match your filters. Loosen them in the sidebar.")
    st.stop()

if weights:
    filtered["score"] = score_companies(filtered, weights, sector_relative)
    filtered = filtered.sort_values("score", ascending=False)
    columns = ["name", "sector", "score"] + [c for c in weights if c not in columns] + columns[2:]
    columns = list(dict.fromkeys(columns))  # remove any duplicates, keep order
else:
    st.info("Move at least one factor slider above 0 to get scores.")

# ---------- Step 6: Page layout ----------
tab_screen, tab_dive = st.tabs(["Screener", "Company deep dive"])

with tab_screen:
    st.caption(f"{len(filtered)} companies match your filters")

    formats = {c: NUMBER_FORMATS[c] for c in columns if c in NUMBER_FORMATS}
    st.dataframe(
        filtered[columns].style.format(formats, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download results as CSV",
        filtered[columns].to_csv(index=False),
        file_name="ftse_screen.csv",
        mime="text/csv",
    )

    if weights:
        st.subheader("Top 10 by score")
        top = filtered.head(10).sort_values("score")
        bar = px.bar(top, x="score", y="name", orientation="h", color="sector",
                     labels={"score": "Score (0-100)", "name": ""})
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("Risk vs return")
    scatter = px.scatter(
        filtered, x="vol_1y", y="ret_12m", color="sector", hover_name="name",
        labels={"vol_1y": "Volatility (1 year)", "ret_12m": "12-month return"},
    )
    scatter.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(scatter, use_container_width=True)

with tab_dive:
    choice = st.selectbox("Pick a company", table["name"])
    row = table[table["name"] == choice].iloc[0]
    history = prices[row["ticker"]].dropna()

    def show(value, fmt):
        return "-" if pd.isna(value) else fmt.format(value)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("12-month return", show(row["ret_12m"], "{:.1%}"))
    c2.metric("Volatility", show(row["vol_1y"], "{:.1%}"))
    c3.metric("Max drawdown", show(row["max_dd"], "{:.1%}"))
    c4.metric("P/E", show(row["pe"], "{:.1f}"))

    chart_data = history.to_frame("Price")
    chart_data["50-day average"] = history.rolling(50).mean()
    chart_data["200-day average"] = history.rolling(200).mean()
    st.line_chart(chart_data)
    st.caption("Prices are as reported by Yahoo Finance (UK shares are usually in pence).")

with st.expander("How the score works"):
    st.write(
        "Each factor is ranked across the companies from 0 (worst) to 1 (best). "
        "For factors where lower is better (P/E, debt, volatility) the ranking is flipped. "
        "The ranks are combined into a weighted average using your slider settings and "
        "scaled to 0-100. Missing data gets a neutral rank of 0.5. Ranks are used instead "
        "of raw values so one extreme number can't dominate the result."
    )