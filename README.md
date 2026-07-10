[README.md](https://github.com/user-attachments/files/29871024/README.1.md)
# Portfolio Risk Analyzer — Fidelity Edition

A portfolio risk management tool built in Python. Upload your Fidelity positions export and get professional-grade risk metrics and a rule-based stock scoring engine — all running on your actual holdings.

Also includes a **standalone ticker analyzer**: type any stock symbol and get a full strategy-inferred report card without uploading anything.

---

## What it does

### Portfolio analysis (upload your Fidelity CSV)
- Parses real Fidelity position exports — handles metadata rows, cash positions, fractional shares, and all the quirks
- Computes professional risk metrics: VaR (95% and 99%), CVaR, Sharpe ratio, annualized volatility, max drawdown, beta, alpha, R-squared vs S&P 500
- Correlation heatmap — shows whether your "diversified" portfolio is actually one concentrated bet
- Covariance-backed rebalancing flags — computes the exact volatility reduction from suggested weight shifts using `w^T · Σ · w`
- Sector breakdown and concentration analysis (HHI index)

### Rule-based stock scoring engine
Automatically infers whether each stock is **Value**, **Growth**, **Momentum**, or **Speculative** based on its fundamentals, then applies the appropriate scoring rules:

| Strategy | Rules applied |
|----------|--------------|
| Value | P/E vs sector average, P/B ratio, debt-to-equity, profit margins, dividend yield, 52-week pullback |
| Growth | Revenue growth YoY, earnings growth, PEG ratio, analyst upside, gross margins, forward vs trailing P/E |
| Momentum | Price vs 200-day MA, price vs 50-day MA, 52-week range percentile, 6-month return vs SPY, beta |
| Speculative | Momentum rules only (no earnings to value) |

Every signal is fully explainable — you can trace exactly which rules fired and why.

### Standalone ticker analyzer
No portfolio upload needed. Type any ticker → strategy auto-inferred → full report card with fundamentals, rule-by-rule score breakdown, and 1-year price chart with 50-day and 200-day moving averages overlaid.

---

## Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation, time series |
| `numpy` | Vectorized math and matrix operations |
| `yfinance` | Free price history and fundamentals (no API key needed) |
| `scipy` | Linear regression for beta/alpha |
| `plotly` | Interactive charts |
| `streamlit` | Web app framework |
| `openpyxl` | Excel file support |

No paid APIs. No LLM. Everything runs locally after the initial data fetch.

---

## Project structure

```
portfolio_analyzer/
├── app.py                  # Streamlit dashboard — main entry point
├── requirements.txt
├── .gitignore
├── data/
│   └── fidelity_export.csv # Your export goes here (not committed)
└── src/
    ├── __init__.py
    ├── parser.py           # Fidelity CSV parser and normalization
    ├── fetcher.py          # yfinance price history and fundamentals
    ├── metrics.py          # All risk calculations
    └── signals.py          # Strategy inference and scoring rules
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/NovaleighEB-dev/portfolio-risk-analyzer.git
cd portfolio-risk-analyzer
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Export your Fidelity positions
In Fidelity: **Accounts → Positions → Download (top right) → CSV**

Rename and place it in the data folder:
```bash
mv ~/Downloads/Portfolio_Positions_*.csv data/fidelity_export.csv
```

### 5. Run
```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

---

## How to use it

**Ticker analyzer** (no upload needed):
- Type any ticker in the sidebar (e.g. `NVDA`, `AAPL`, `WMT`)
- Click **Analyze ticker**
- Get the full strategy-inferred report card instantly

**Full portfolio analysis**:
- Upload your Fidelity CSV in the sidebar
- Click **Run portfolio analysis**
- Navigate the four tabs: Overview, Risk Analysis, Rebalancing, Stock Insights

---

## Risk metrics explained

| Metric | What it means |
|--------|--------------|
| **VaR 95%** | On 95% of days, your loss will be less than this amount |
| **CVaR 95%** | Average loss on the worst 5% of days — always worse than VaR |
| **Sharpe ratio** | Excess return per unit of risk. Above 1.0 is good, above 2.0 is exceptional |
| **Max drawdown** | Largest peak-to-trough decline in the historical window |
| **Beta** | How much the portfolio moves relative to S&P 500. 1.5 = amplified, 0.6 = dampened |
| **Alpha** | Return not explained by market exposure — positive means you beat the market risk-adjusted |
| **HHI** | Herfindahl-Hirschman Index for concentration. Near 0 = diversified, near 1 = concentrated |

---

## The math

Portfolio variance is computed using the full covariance matrix:

```
σ²_portfolio = w^T · Σ · w
```

Where `w` is the weight vector and `Σ` is the annualized covariance matrix. This accounts for all pairwise correlations and is the reason a "diversified" 15-stock portfolio can still behave like a single concentrated bet.

---

## Disclaimer

This tool is for educational and research purposes only. Nothing here is financial advice. All signals and metrics are based on historical data and statistical models — past performance does not guarantee future results. Always do your own research before making investment decisions.

---

## Author

Built by Novaleigh as a learning project combining quantitative finance, data engineering, and software architecture.
