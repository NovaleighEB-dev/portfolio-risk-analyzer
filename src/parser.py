import pandas as pd 
import numpy as np 
import pathlib as Path 

SKIP_TICKERS = {'FCASH', 'FDRXX', 'SPAXX', 'FZFXX', 'FMPXX', 'Pending Activity', '', 'CORE'}
# Skip those tickers b/c there's no point in "analyzing" them esp. with yfinance 
COMMON_ETFS = {
 'VOO','VTI','SPY','QQQ','IWM','VEA','VWO','BND','AGG',
 'GLD','SLV','VNQ','XLK','XLF','XLE','XLV','ARKK','SCHD',
 'VIG','VXUS','IEMG','EFA','TLT','SHY','HYG','LQD',
}

# Finding the rows that contain the important tickers
def find_header_row(filepath: str) -> int: 
    with open(filepath, 'r', encoding='utf-8-sig') as f: 
        for i, line in enumerate(f): 
            if 'Symbol' in line and 'Quantity' in line: 
                return i 
    raise ValueError(
        'Could not find data headers in this file'
    )

# Clean numeric values 
# Remove $, convert percentages, remove plus, N/A to NaN 
def clean_numeric(value) -> float: 
    if pd.isna(value): return np.nan 
    s = str(value).strip() 
    if s in ('N/A', 'N/a' '--', '', 'nan'): return np.nan 
    s = s.replace('$', '').replace(',', '').replace('%', '')
    try: return float(s) 
    except ValueError: return np.nan

def parse_fidelity(filepath: str) -> pd.DataFrame: 
    header_row = find_header_row(filepath)
    df = pd.read_csv(filepath, skiprows=header_row, index_col=False)

    df.columns = df.columns.str.strip().str.replace('"', '')
    df = df.map(lambda x : x.strip() if isinstance(x, str) else x)

    # Skip and remove a bunch of stuff from the data like **, rows with no symbols
    df = df[df['Symbol'].notna()]
    df['Symbol'] = df['Symbol'].str.replace('*', '', regex = False).str.strip()
    df = df[~df['Symbol'].str.strip().isin(SKIP_TICKERS)]
    df = df[df['Symbol'].str.match(r'^[A-Z]{1,5}$', na = False)]

    result = pd.DataFrame({
        'ticker': df['Symbol'].str.strip(), 
        'shares': df['Quantity'].apply(clean_numeric), 
        'avg_cost': df['Average Cost Basis'].apply(clean_numeric), 
        'current_price': df['Last Price'].apply(clean_numeric), 
        'current_value': df['Current Value'].apply(clean_numeric)
    })

    result = result.dropna(subset=['ticker', 'shares', 'current_value'])
    result = result[result['shares'] > 0] 
    result = result.reset_index(drop=True) 

    return result 

def load_portfolio(filepath: str) -> pd.DataFrame: 
    df = parse_fidelity(filepath) 
    if len(df) == 0: 
        raise ValueError('No Valid Positions Found!')
    
    # Create portfolio weight 
    total = df['current_value'].sum() 
    df['weight'] = df['current_value'] / total 

    # Create Asset Type: either ETF or stock 
    df['asset_type'] = df['ticker'].apply(
        lambda t: 'ETF' if t in COMMON_ETFS else 'stock'
    )

    # Create unrealized return: (current price / avg_cost) - 1 
    df['unrealized_pct'] = (df['current_price'] / df['avg_cost'] - 1).round(4) 

    print(f'Loaded {len(df)} positions | Total Value: ${total:,.2f}')
    etf_count = (df['asset_type'] == 'ETF').sum() 
    stock_count = (df['asset_type'] == 'stock').sum() 
    print(f' {stock_count} Stocks   |   {etf_count} ETFS')

    return df