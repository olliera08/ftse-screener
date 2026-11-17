FTSE 100 Screener

A Python web app that ranks FTSE 100 companies using a weighted score of value (P/E), quality (ROE), momentum (12-month return) and low volatility.

Built with Python, pandas, yfinance, Streamlit and Plotly.

## How to run
1. Install Python
2. Run: python -m pip install -r requirements.txt
3. Run: python -m streamlit run app.py

## How the scoring works
Each metric is ranked from 0 to 1 across companies, "lower is better" metrics are flipped, and the ranks are combined using the slider weights. Missing data gets a neutral score.

## Limitations
Data comes from Yahoo Finance through yfinance, an unofficial source, so it can be missing or wrong. Educational project, not financial advice.
