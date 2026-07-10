import yfinance as yf 
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta

# Download daily adjusted closing prices for the list of tickers 
# Use a period of 2 years b/c that's industry standard for medium-term risk 
# You don't want too much or too little 
def fetch_price_history(tickers: list, years: int = 1) -> pd.DataFrame:
    end = datetime.today() 
    start = end - timedelta(days = 365 * years) 

    raw = yf.download(
        tickers, 
        start = start.strftime('%Y-%m-%d'),
        end = end.strftime('%Y-%m-%d'),
        auto_adjust = True, 
        progress = False 
    )

    # yfinance returns MultiIndex columns when multiple tickers 
    prices = raw['Close'] if len(tickers) > 1 else raw[['Close']].rename(
        columns = {'Close': tickers[0]}
    )
    # Remove empty rows and fill in the gaps 
    prices = prices.dropna(how = 'all')
    prices = prices.ffill()

    # Handle missing price data 
    missing = [t for t in tickers if t not in prices.columns]
    if missing: 
        print('These tickers will be excluded from analysis due to no price data found:')
        print(f'{missing}')

    return prices 

# get the fundamental data for each stock with yfinance
def fetch_fundamentals(ticker: str) -> dict: 
    try: 
        info = yf.Ticker(ticker).info 
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'pe_ratio': info.get('trailingPE'),
            'peg_ratio': info.get('pegRatio'),
            'pb_ratio': info.get('priceToBook'),
            'market_cap': info.get('marketCap'),
            'analyst_rating': info.get('recommendationKey', 'none'),
            '52w_high': info.get('fiftyTwoWeekHigh'),
            '52w_low': info.get('fiftyTwoWeekLow'),
            'dividend_yield': info.get('dividendYield'),
            'debt_to_equity': info.get('debtToEquity'),
            'beta': info.get('beta'),
            'profit_margins': info.get('profitMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'target_price': info.get('targetMeanPrice'), 
            'current_price': info.get('currentPrice'),
            '52w_range_pct': (info.get('fiftyTwoWeekHigh') - info.get('fiftyTwoWeekLow')) / info.get('fiftyTwoWeekLow') 
            if info.get('fiftyTwoWeekHigh') and info.get('fiftyTwoWeekLow') else None,
    }
    except Exception: 
        return {'ticker': ticker, 'name': ticker, 'sector': 'Unknown'}
    

def enrich_positions(positions: pd.DataFrame) -> pd.DataFrame: 
    sectors = [] 
    for _, row in positions.iterrows(): 
        if row['asset_type'] == 'ETF': 
            sectors.append('ETF')
        else: 
            info = fetch_fundamentals(row['ticker'])
            sectors.append(info.get('sector', 'Unknown'))
    positions = positions.copy() 
    positions['sector'] = sectors 
    return positions 