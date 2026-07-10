import numpy as np 
import pandas as pd 
from scipy import stats 
from src.fetcher import fetch_price_history

TRADING_DAYS = 252 # trading days per year for US market 

# Compute returns using log (better compared to simple return) 
# log(P_t / P_t-1)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame: 
    returns = np.log(prices / prices.shift(1))
    return returns.dropna() # would drop the first value since no prev. day 

# Calculate daily portoflio return as a weighted sum of individual returns. 
# sum(w_i * r_i_t) => 1.0 
def portfolio_returns(returns: pd.DataFrame, weighted: np.ndarray) -> pd.Series: 
    return returns.dot(weighted)

# multiply by 252 for annualized return 
def compute_annual_return(daily_returns: pd.Series) -> float: 
    return float(daily_returns.mean() * TRADING_DAYS)

# Calculate annualized volatility 
def compute_annualized_volatility(daily_returns: pd.Series) -> float: 
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))

# Calculating 30-day rolling volatility 
def computer_rolling_volatility(daily_returns: pd.Series) -> pd.Series: 
    return daily_returns.rolling(30).std() * np.sqrt(TRADING_DAYS)

# Calculating Sharpe Ratio 
# We're just gonna use risk-free rate as 0.5 for now by default
def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.05) -> float: 
    excess = daily_returns - (risk_free_rate/TRADING_DAYS)
    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) # annualized 
    return sharpe 

# Calculate maximum_drawdown 
def max_drawdown(daily_returns: pd.Series) -> float: 
    Wt = np.exp(daily_returns.cumsum())
    peakt = Wt.cummax()
    DDt = (Wt - peakt) / peakt # drawdown 
    return float(DDt.min()) 

# Calculate drawdown time series for cahrting 
def drawdown_series(daily_returns: pd.Series) -> pd.Series: 
    Wt = np.exp(daily_returns.cumsum())
    peakt = Wt.cummax() 
    return (Wt - peakt) / peakt

# Calculate historical VaR at given confidence level (default is 95%) 
def historical_VaR(daily_returns: pd.Series, confidence: float = 0.95) -> float: 
    cutoff = np.percentile(daily_returns, (1-confidence) * 100) 
    return abs(float(cutoff))

# Calculate parametruc VaR, assuming normal distribution 
# also just to compare 
def parametric_VaR(daily_returns: pd.Series, confidence: float = 0.95) -> float: 
    mu = daily_returns.mean() 
    std = daily_returns.std() 
    z = stats.norm.ppf(1-confidence) 
    return abs(float(mu + z * std))

# converting var to sollars
def var_to_dollars(var_pct: float, portfolio_value: float) -> float: 
    return var_pct * portfolio_value

# Calculate CVaR, average loss on days wrose than the VaR threshold 
def cvar(daily_returns: pd.Series, confidence: float = 0.95) -> float: 
    cutoff = np.percentile(daily_returns, (1-confidence)*100) 
    tail_returns = daily_returns[daily_returns <= cutoff] 
    return abs(float(tail_returns.mean()))

 # calculate beta and alpha vs. the market and use linear regression 
def compute_beta(portfolio_returns: pd.Series, market_ticker: str = 'SPY') -> dict:
    mkt_prices = fetch_price_history([market_ticker])
    mkt_returns = np.log(mkt_prices / mkt_prices.shift(1)).dropna() 
    mkt_returns = mkt_returns[market_ticker]
    common = portfolio_returns.index.intersection(mkt_returns.index)
    p = portfolio_returns.loc[common].values 
    m = mkt_returns.loc[common].values
    p = np.array(p, dtype=float).flatten()
    m = np.array(m, dtype=float).flatten()

    # y = alpha + beta * x 
    beta, alpha, r_val, _, _ = stats.linregress(m, p)
    return{
        'beta': round(float(beta), 3), 
        'alpha': round(float(alpha * TRADING_DAYS), 4), # annualized 
        'r_squared': round(float(r_val ** 2), 3) 
    }

# returns the pearson correlation matrx between all assets
def correlation_mattrix(returns: pd.DataFrame) -> pd.DataFrame: 
    return returns.corr() 

# returns the annualized covariance matrix 
def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame: 
    return returns.cov() * TRADING_DAYS 

# Prtoflio variance using covariance matrix: 
# sigma^2 = w^T * sigma * w 
# this accounts for all pairwise correlations 
def portfolio_variance(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return float(weights @ cov_matrix @ weights)

# annualized portfolio volatility from covariance matrix 
def portfolio_vol_cov(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float: 
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))

# Measure portfolio concentration 
# HHI = sum of squared weights 

def concentration_risk(positions: pd.DataFrame) -> dict: 
    w = positions['weight'].values 
    hhi = float(np.sum(w**2))

    top5 = positions.nlargest(5, 'weight')[
        ['ticker', 'weight', 'current_value', 'asset_type']
    ]

    return {
        'hhi': hhi, 
        'top5': top5, 
        'top5_pct': float(top5['weight'].sum()), 
        'n_positions': len(positions)
    }

def sector_breakdown(positions: pd.DataFrame) -> pd.DataFrame: 
    return(
        positions.groupby('sector')
        .agg(weight = ('weight', 'sum'), value = ('current_value', 'sum'))
        .sort_values('weight', ascending=False)
        .reset_index()
    )

# Rebalancing which is questionable for me but do it for the math 
def rebalancing_flags(positions: pd.DataFrame, cov_matrix: pd.DataFrame, threshold: float = 0.2) -> list: 
    ticker = positions['ticker'].tolist() 
    weights = positions['weight'].values.copy() 
    n = len(weights) 
    flags = [] 

    current_vol = float(np.sqrt(weights @ cov_matrix @ weights))

    for i, row in positions.iterrows():
        if row['weight'] <= threshold: 
            continue 

        ticker = row['ticker']
        excess = row['weight'] - threshold

        new_w = weights.copy() 
        new_w[i] = threshold
        others = [j for j in range(n) if j != i]
        new_w[others] += excess / len(others)

        new_vol = float(np.sqrt(new_w @ cov_matrix.values @ new_w))
        vol_reduction = current_vol - new_vol
        flags.append({
            'ticker': ticker,
            'current_weight': row['weight'],
            'suggested_weight': threshold,
            'current_vol': current_vol,
            'new_vol': new_vol,
            'vol_reduction': vol_reduction,
            'message': (
            f"Reducing {ticker} from {row['weight']:.1%} to {threshold:.1%} "
            f"would lower portfolio volatility from {current_vol:.1%} "
            f"to {new_vol:.1%} (a {vol_reduction:.1%} improvement"), 
        })
        return flags
    
def run_all_metrics(positions: pd.DataFrame, prices: pd.DataFrame) -> dict:
    tickers = [t for t in positions['ticker'].tolist() if t in prices.columns]
    weights = positions.set_index('ticker').loc[tickers, 'weight'].values
    weights = weights / weights.sum() # renormalize after any drops
    prices = prices[tickers].dropna()
    returns = compute_returns(prices)
    port_r = portfolio_returns(returns, weights)
    cov_mat = covariance_matrix(returns)
    total = float(positions['current_value'].sum())
    var95 = historical_VaR(port_r, 0.95)
    var99 = historical_VaR(port_r, 0.99)
    return {
        'total_value': total,
        'annualized_return': compute_annual_return(port_r),
        'volatility': compute_annualized_volatility(port_r),
        'sharpe_ratio': calculate_sharpe_ratio(port_r),
        'max_drawdown': max_drawdown(port_r),
        'var_95_pct': var95,
        'var_99_pct': var99,
        'var_95_usd': var_to_dollars(var95, total),
        'cvar_95': cvar(port_r, 0.95),
        'beta': compute_beta(port_r),
        'corr_matrix': correlation_mattrix(returns),
        'cov_matrix': cov_mat,
        'concentration': concentration_risk(positions),
        'sector_breakdown': sector_breakdown(positions),
        'rebalancing_flags': rebalancing_flags(positions, cov_mat),
        'portfolio_returns': port_r,
        'individual_returns':returns,
        'drawdown_series': drawdown_series(port_r), 
    }
