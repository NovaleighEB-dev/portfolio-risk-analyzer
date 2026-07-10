import numpy as np 
import pandas as pd 
import yfinance as yf 
from datetime import datetime, timedelta
from src.fetcher import fetch_fundamentals

# Guess a strategy between value, growth, or momentum based 
def pick_strategy(ticker: str, fundamentals: dict, prices: pd.DataFrame) -> str: 
    pe = fundamentals.get('pe_ratio')
    rev_g = fundamentals.get('revenue_growth') 
    eps_g = fundamentals.get('earnings_growth')

    # get price series for ticker 
    if ticker not in prices.columns: 
        return 'Speculative' 
    price_series = prices[ticker].dropna() 

    # Compute moving averages
    ma50 = price_series.rolling(50).mean().iloc[-1]
    ma200 = price_series.rolling(200).mean().iloc[-1]
    current_price = price_series.iloc[-1]

    # calculate 6 month relative performance vs. spy 
    six_months_ago = price_series.index[-1] - pd.DateOffset(months=6)
    mask = price_series.index >= six_months_ago
    if mask.sum() > 20:
        stock_6m = (price_series[mask].iloc[-1] / price_series[mask].iloc[0]) - 1
    else:
        stock_6m = 0

    # SPY 6-month return for comparison
    if 'SPY' in prices.columns:
        spy = prices['SPY'].dropna()
        spy_mask = spy.index >= six_months_ago
        spy_6m = (spy[spy_mask].iloc[-1] / spy[spy_mask].iloc[0]) - 1
    else: 
        spy_6m = 0.08 # fallback 

    # Picking a strategy 
    # No earnings = speculative unless strong momentum
    if pe is None or pe < 0:
        if current_price > ma200 and stock_6m > spy_6m:
            return 'Momentum'
        return 'Speculative'
    
    # Strong momentum signal overrides everything
    in_uptrend = current_price > ma200
    beating_market = stock_6m > spy_6m
    if in_uptrend and beating_market and stock_6m > 0.15:
        return 'Momentum'
    
    # Growth signal
    high_rev_growth = rev_g is not None and rev_g > 0.15
    high_eps_growth = eps_g is not None and eps_g > 0.20
    if high_rev_growth or high_eps_growth:
        return 'Growth'
    # Value fallback
    return 'Value'

# Getting all the fundamental data for scoring 
def fetch_enriched_fundamentals(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Price history for moving averages
        hist = stock.history(period='1y')
        price_series = hist['Close'] if not hist.empty else pd.Series()
        # Moving averages
        ma50 = price_series.rolling(50).mean().iloc[-1] if len(price_series) >= 50 else None
        ma200 = price_series.rolling(200).mean().iloc[-1] if len(price_series) >= 200 else None
        # 52-week range percentile
        high_52w = info.get('fiftyTwoWeekHigh')
        low_52w = info.get('fiftyTwoWeekLow')
        current = info.get('currentPrice') or info.get('regularMarketPrice')
        if high_52w and low_52w and high_52w != low_52w:
            range_pct = (current - low_52w) / (high_52w - low_52w)
        else:
            range_pct = None
        return {
        'ticker': ticker,
        'name': info.get('longName', ticker),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        # Valuation
        'pe_ratio': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'pb_ratio': info.get('priceToBook'),
        'peg_ratio': info.get('pegRatio'),
        'ev_to_ebitda': info.get('enterpriseToEbitda'),
        # Growth
        'revenue_growth': info.get('revenueGrowth'),
        'earnings_growth': info.get('earningsGrowth'),
        'eps_trailing': info.get('trailingEps'),
        'eps_forward': info.get('forwardEps'),
        # Quantity 
        'profit_margins': info.get('profitMargins'),
        'gross_margins': info.get('grossMargins'),
        'debt_to_equity': info.get('debtToEquity'),
        'current_ratio': info.get('currentRatio'),
        'roe': info.get('returnOnEquity'),
        # Income
        'dividend_yield': info.get('dividendYield'),
        # Price targets
        'target_price': info.get('targetMeanPrice'),
        'analyst_rating': info.get('recommendationKey', 'none'),
        'n_analysts': info.get('numberOfAnalystOpinions', 0),
        # Technicals (computed)
        'current_price': current,
        'ma50': float(ma50) if ma50 else None,
        'ma200': float(ma200) if ma200 else None,
        '52w_high': high_52w,
        '52w_low': low_52w,
        '52w_range_pct': range_pct,
        'beta': info.get('beta'),
        'market_cap': info.get('marketCap'),
        }
    except Exception as e:
        print(f'Warning: could not fetch fundamentals for {ticker}: {e}')
        return {'ticker': ticker, 'name': ticker, 'sector': 'Unknown'}
    
# Build a scoring system based on signals that are considered "good" or "positive"

# PE averages for each sector 
SECTOR_PE_AVERAGES = {
    'Technology': 35,
    'Healthcare': 22,
    'Consumer Cyclical': 20,
    'Consumer Defensive': 18,
    'Industrials': 20,
    'Financial Services': 13,
    'Energy': 12,
    'Utilities': 15,
    'Real Estate': 25,
    'Communication Services': 20,
    'Basic Materials': 16,
    'Unknown': 18,
}

# this is for value strategy so based on the rules for value investing
def score_value(fund: dict, position: dict) -> dict: 
    score = 0
    breakdown = []
    sector = fund.get('sector', 'Unknown')
    sector_pe = SECTOR_PE_AVERAGES.get(sector, 18)
    # 1) P/E vs sector average
    pe = fund.get('pe_ratio')
    if pe and pe > 0:
        if pe < sector_pe * 0.80:
            score += 1
            breakdown.append(f'#1: +1 P/E {pe:.1f} is 20%+ below sector avg ({sector_pe})')
        elif pe > sector_pe * 1.50:
            score -= 1
            breakdown.append(f'#1: -1 P/E {pe:.1f} is 50%+ above sector avg ({sector_pe})')
        else:
            breakdown.append(f'#1: 0 P/E {pe:.1f} is near sector avg ({sector_pe})')
    else:
        breakdown.append('V1 0 P/E not available')

    # 2) Price-to-Book
    pb = yf.Ticker(fund.get('ticker')).info.get('priceToBook')
    if pb:
        if pb < 2.0: # even though buffet prefers 1.5
            score += 1
            breakdown.append(f'#2: +1 P/B {pb:.2f} — below 2.0 (value territory)')
        elif pb > 4.0:
            score -= 1
            breakdown.append(f'#2: -1 P/B {pb:.2f} — above 4.0 (expensive assets)')
        else:
            breakdown.append(f'#2: 0 P/B {pb:.2f} — moderate')
    else:
        breakdown.append('V2 0 P/B not available')

    # #3: Debt-to-Equity
    de = fund.get('debt_to_equity')
    if de is not None:
        if de < 100: # yfinance returns D/E * 100
            score += 1
            breakdown.append(f'#3: +1 D/E {de/100:.2f} — low debt (below 1.0)')
        elif de > 200:
            score -= 1
            breakdown.append(f'#3: -1 D/E {de/100:.2f} — high debt (above 2.0)')
        else:
            breakdown.append(f'#3: 0 D/E {de/100:.2f} — moderate debt')
    else:
        breakdown.append('#3: 0 D/E not available')
    # #4: Profit margin
    margin = fund.get('profit_margins')
    if margin is not None:
        if margin > 0.10:
            score += 1
            breakdown.append(f'#4 +1 Profit margin {margin:.1%} — above 10%')
        elif margin < 0:
            score -= 1
            breakdown.append(f'#4: -1 Profit margin {margin:.1%} — losing money')
        else:
            breakdown.append(f'#4: 0 Profit margin {margin:.1%} — thin but positive')
    else:
        breakdown.append('#4: 0 Profit margin not available') 

    # #5: Divident yield 
    div = fund.get('dividend_yield')
    if div and div > 0.015: 
        score += 1
        breakdown.append(f'#5: +1 Div yield {div:.2%} - above 1.5%')
    else: 
        breakdown.append(f'#5: 0 Dividend yield {div:.2%}' if div else 'V5 0 No dividend')
    
    # 6: Distance from 52-week high
    high = fund.get('52w_high')
    curr = fund.get('current_price')
    if high and curr:
        pct_from_high = (curr / high) - 1
        if pct_from_high < -0.20:
            score += 1
            breakdown.append(f'#6: +1 {pct_from_high:.1%} below 52-week high — pullback opportunity')
        elif pct_from_high > -0.05:
            breakdown.append(f'#6: 0 Near 52-week high ({pct_from_high:.1%}) — limited upside')
        else:
            breakdown.append(f'#6: 0 {pct_from_high:.1%} from 52-week high')
    else:
        breakdown.append('V6 0 Price data not available')
    return {'score': score, 'max_score': 6, 'breakdown': breakdown}

# Calculate score on growth investing 
def score_growth(fund: dict, position: dict) -> dict:
    score = 0
    breakdown = []
    # G1: Revenue growth
    rev_g = fund.get('revenue_growth')
    if rev_g is not None:
        if rev_g > 0.25:
            score += 2
            breakdown.append(f'G1: +2 Revenue growth {rev_g:.1%} — exceptional (>25%)')
        elif rev_g > 0.15:
            score += 1
            breakdown.append(f'G1: +1 Revenue growth {rev_g:.1%} — solid (>15%)')
        elif rev_g < 0:
            score -= 1
            breakdown.append(f'G1: -1 Revenue DECLINING {rev_g:.1%} — red flag')
        else:
            breakdown.append(f'G1: 0 Revenue growth {rev_g:.1%} — below growth threshold')
    else:
        breakdown.append('G1 0 Revenue growth not available')
    # G2: Earnings growth
    eps_g = fund.get('earnings_growth')
    if eps_g is not None:
        if eps_g > 0.20:
            score += 1
            breakdown.append(f'G2: +1 Earnings growth {eps_g:.1%} — above 20%')
        elif eps_g < 0:
            score -= 1
            breakdown.append(f'G2: -1 Earnings SHRINKING {eps_g:.1%}')
        else:
            breakdown.append(f'G2: 0 Earnings growth {eps_g:.1%} — moderate')
    else:
        breakdown.append('G2: 0 Earnings growth not available')
    # G3: PEG ratio
    peg = fund.get('peg_ratio')
    if peg and peg > 0:
        if peg < 1.5:
            score += 1
            breakdown.append(f'G3: +1 PEG {peg:.2f} — reasonable price for growth')
        elif peg > 3.0:
            score -= 1
            breakdown.append(f'G3: -1 PEG {peg:.2f} — overpaying for growth')
        else:
            breakdown.append(f'G3: 0 PEG {peg:.2f} — moderate')
    else:
        breakdown.append('G3: 0 PEG not available')
    # G4: Analyst price target upside
    target = fund.get('target_price')
    curr = fund.get('current_price')
    if target and curr:
        upside = (target / curr) - 1
        if upside > 0.15:
            score += 1
            breakdown.append(f'G4: +1 Analyst target ${target:.2f} = {upside:.1%} upside')
        elif upside < -0.10:
            score -= 1
            breakdown.append(f'G4: -1 Analyst target ${target:.2f} = {upside:.1%} downside')
        else:
            breakdown.append(f'G4: 0 Analyst target ${target:.2f} = {upside:.1%} upside')
    else:
        breakdown.append('G4: 0 No analyst target available')
    # G5: Gross margin
    gm = fund.get('gross_margins')
    if gm is not None:
        if gm > 0.40:
            score += 1
            breakdown.append(f'G5: +1 Gross margin {gm:.1%} — high margin business')
        else:
            breakdown.append(f'G5: 0 Gross margin {gm:.1%} — capital intensive')
    else:
        breakdown.append('G5: 0 Gross margin not available')
    # G6: Forward vs trailing P/E
    pe_trail = fund.get('pe_ratio')
    pe_forward = fund.get('forward_pe')
    if pe_trail and pe_forward and pe_trail > 0 and pe_forward > 0:
        if pe_forward < pe_trail * 0.85:
            score += 1
            breakdown.append(f'G6: +1 Forward P/E {pe_forward:.1f} < Trailing {pe_trail:.1f} - earnings accelerating')
        else:
            breakdown.append(f'G6: 0 Forward P/E {pe_forward:.1f} vs Trailing {pe_trail:.1f}')
    else:
        breakdown.append('G6: 0 P/E data incomplete')
    
    return {'score': score, 'max_score': 7, 'breakdown': breakdown}



# Now for momentum strategy 
def score_momentum(fund: dict, position: dict, prices: pd.DataFrame, spy_6m_return: float) -> dict:
    score = 0
    breakdown = []
    ticker = fund['ticker']
    curr = fund.get('current_price')
    ma50 = fund.get('ma50')
    ma200 = fund.get('ma200')
    # M1: Price vs 200-day MA
    if curr and ma200:
        if curr > ma200:
            score += 2
            breakdown.append(f'M1: +2 ${curr:.2f} above 200-day MA ${ma200:.2f} — long-term uptrend')
        else:
            score -= 2
            breakdown.append(f'M1: -2 ${curr:.2f} below 200-day MA ${ma200:.2f} — long-term downtrend')
    else:
        breakdown.append('M1: 0 Moving average data not available')
    # M2: Price vs 50-day MA
    if curr and ma50:
        if curr > ma50:
            score += 1
            breakdown.append(f'M2: +1 Above 50-day MA ${ma50:.2f} — short-term uptrend')
        else:
            score -= 1
            breakdown.append(f'M2: -1 Below 50-day MA ${ma50:.2f} — short-term downtrend')
    else:
        breakdown.append('M2: 0 50-day MA not available')
    # M3: 52-week range percentile
    rng = fund.get('52w_range_pct')
    if rng is not None:
        if rng > 0.75:
            score += 1
            breakdown.append(f'M3: +1 In top 25% of 52-week range ({rng:.0%}) — price strength')
        elif rng < 0.25:
            score -= 1
            breakdown.append(f'M3 -1 In bottom 25% of 52-week range ({rng:.0%}) — weak')
        else:
            breakdown.append(f'M3 0 52-week range position: {rng:.0%}')
    else:
        breakdown.append('M3 0 Range data not available')

    # M4: 6-month return vs SPY
    if ticker in prices.columns:
        p = prices[ticker].dropna()
        six_m_ago = p.index[-1] - pd.DateOffset(months=6)
        p6 = p[p.index >= six_m_ago]
        if len(p6) > 20:
            stock_6m = (p6.iloc[-1] / p6.iloc[0]) - 1
            vs_spy = stock_6m - spy_6m_return
            if vs_spy > 0:
                score += 1
                breakdown.append(f'M4: +1 6m return {stock_6m:.1%} vs SPY {spy_6m_return:.1%} — beating market by {vs_spy:.1%}')
            elif vs_spy < -0.10:
                score -= 1
                breakdown.append(f'M4: -1 6m return {stock_6m:.1%} vs SPY {spy_6m_return:.1%} — lagging by {abs(vs_spy):.1%}')
            else:
                breakdown.append(f'M4: 0 6m return {stock_6m:.1%} vs SPY {spy_6m_return:.1%}')
        else:
            breakdown.append('M4: 0 Insufficient price history for 6m return')
    else:
        breakdown.append('M4: 0 Ticker not in price history')

    # M5: Beta
    beta = fund.get('beta')
    if beta:
        if beta > 1.2:
            score += 1
            breakdown.append(f'M5: +1 Beta {beta:.2f} — amplified market moves')
        elif beta < 0.7:
            score -= 1
            breakdown.append(f'M5: -1 Beta {beta:.2f} — low beta dampens momentum')
        else:
            breakdown.append(f'M5: 0 Beta {beta:.2f} — moderate')
    else:
        breakdown.append('M5: 0 Beta not available')

    return {'score': score, 'max_score': 6, 'breakdown': breakdown}

# Check how well the position would fit into the portfolio regardless of strategy 
def score_portfolio_fit(position: dict, metrics: dict) -> dict:
    score = 0
    breakdown = []
    ticker = position['ticker']
    # PF1: Concentration risk
    weight = position['weight']
    if weight > 0.25:
        score -= 2
        breakdown.append(f'PF1: -2 Position is {weight:.1%} of portfolio — dangerously overweight')
    elif weight > 0.15:
        score -= 1
        breakdown.append(f'PF1: -1 Position is {weight:.1%} of portfolio — overweight')
    else:
        breakdown.append(f'PF1: 0 Position weight {weight:.1%} — acceptable')
    
    # PF2: Correlation with rest of portfolio
    corr_matrix = metrics.get('corr_matrix')
    if corr_matrix is not None and ticker in corr_matrix.columns:
        others = [c for c in corr_matrix.columns if c != ticker]
        avg_corr = corr_matrix.loc[ticker, others].mean()
        if avg_corr > 0.80:
            score -= 1
            breakdown.append(f'PF2: -1 Avg correlation with portfolio: {avg_corr:.2f} — highly correlated, low diversification benefit')
        elif avg_corr < 0.40:
            score += 1
            breakdown.append(f'PF2: +1 Avg correlation with portfolio: {avg_corr:.2f} — low correlation, good diversifier')
        else:
            breakdown.append(f'PF2: 0 Avg correlation with portfolio: {avg_corr:.2f}')
    else:
        breakdown.append('PF2: 0 Correlation data not available')
    return {'score': score, 'max_score': 0, 'breakdown': breakdown}

# Compute SPY's 6 month return for rel. strength comparison
def compute_spy_6m_return(prices: pd.DataFrame) -> float:
    if 'SPY' not in prices.columns:
        return 0.08 # fallback assumption
    spy = prices['SPY'].dropna()
    six_m_ago = spy.index[-1] - pd.DateOffset(months=6)
    p6 = spy[spy.index >= six_m_ago]
    if len(p6) < 20:
        return 0.08
    return float((p6.iloc[-1] / p6.iloc[0]) - 1)


# Generate report card from everything for one stock 
# this includes picking the strategy, score, then return signal
def generate_report_card(position: dict, fund: dict, metrics: dict, prices: pd.DataFrame, spy_6m: float) -> dict:
    ticker = position['ticker']
    strategy = pick_strategy(ticker, fund, prices)
    # Run strategy-specific scoring
    if strategy == 'Value':
        strat_result = score_value(fund, position)
    elif strategy == 'Growth':
        strat_result = score_growth(fund, position)
    elif strategy == 'Momentum':
        strat_result = score_momentum(fund, position, prices, spy_6m)
    else: # If Speculative then just use momentum
        strat_result = score_momentum(fund, position, prices, spy_6m)
        strategy = 'Speculative'
    # Portfolio fit rules (always run)
    fit_result = score_portfolio_fit(position, metrics)
    # Total score
    total_score = strat_result['score'] + fit_result['score']
    max_score = strat_result['max_score'] 
    # Map score to signal
    pct = total_score / max_score if max_score > 0 else 0
    if pct >= 0.40:
        signal = 'BUY'
    elif pct <= -0.20 or total_score <= -2:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    # Full breakdown
    all_rules = strat_result['breakdown'] + fit_result['breakdown']
    return {
        'ticker': ticker,
        'strategy': strategy,
        'signal': signal,
        'total_score': total_score,
        'max_score': max_score,
        'score_pct': pct,
        'rules': all_rules,
        'fundamentals': fund,
        'position': position,
        'unrealized_pct': position.get('unrealized_pct', 0),
    }

# Run report cards for every position
def run_all_signals(positions: pd.DataFrame, metrics: dict, prices: pd.DataFrame) -> list:
    spy_6m = compute_spy_6m_return(prices)
    cards = []
    for _, row in positions.iterrows():
        fund = fetch_enriched_fundamentals(row['ticker'])
        card = generate_report_card(row.to_dict(), fund, metrics, prices, spy_6m)
        cards.append(card)
    return cards