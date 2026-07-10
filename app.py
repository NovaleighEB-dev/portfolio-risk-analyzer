from src.parser import load_portfolio, parse_fidelity
from src.fetcher import fetch_price_history, fetch_fundamentals, enrich_positions
from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go 
import plotly.express as px 
import tempfile, os
from src.parser import load_portfolio
from src.fetcher import fetch_price_history, enrich_positions, fetch_fundamentals
from src.metrics import run_all_metrics
from src.signals import run_all_signals, compute_spy_6m_return, generate_report_card, pick_strategy

SIGNAL_COLORS = {'BUY': '#1D9E75', 'HOLD': '#BA7517', 'SELL': '#D85A30'}
STRATEGY_COLORS = {
    'Value': '#1D9E75', 'Growth': '#7c6ae6',
    'Momentum': '#BA7517', 'Speculative': '#D85A30',
}
st.set_page_config(
    page_title = 'Portfolio Risk Analyzer',
    page_icon = ':bar_chart:',
    layout = 'wide',
)

def _render_report_card(card: dict, fund: dict, standalone: bool = False):
        ticker = card['ticker']
        signal = card['signal']
        strategy = card['strategy']
        score = card['total_score']
        max_sc = card['max_score']
        sig_color = SIGNAL_COLORS.get(signal, '#888')
        strat_color = STRATEGY_COLORS.get(strategy, '#888')
        # Row 1: signal badge | strategy badge | score bar
        col1, col2, col3 = st.columns([0.15, 0.20, 0.65])
        style = "background:#7c2d2d;color:white;border-radius:10px;padding:16px 24px;text-align:center;" \
        "font-family:monospace;font-size:18px;font-weight:520;width:100%;"
        sstyle = "background:#1D9E75;color:white;border-radius:10px;padding:16px 24px;text-align:center;" \
        "font-family:monospace;font-size:18px;font-weight:520;width:100%;"
        with col1:
            st.markdown(f'<div style="{style}">{signal} </div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="{sstyle}">{strategy} </div>', unsafe_allow_html=True)
        with col3:
            pct = (score / max_sc) if max_sc > 0 else 0
            st.markdown(f'**Score: {score} / {max_sc}** ({pct:.0%})')
            st.progress(max(0.0, min(1.0, (pct + 1) / 2)))
        st.markdown('---')
        # Fundamentals grid (2 rows of 4 metrics each)
        st.markdown('#### Key metrics')
        f1, f2, f3, f4 = st.columns(4)
        pe = fund.get('pe_ratio')
        pb = yf.Ticker(fund.get('ticker')).info.get('priceToBook')
        rg = fund.get('revenue_growth')
        pm = fund.get('profit_margins')
        f1.metric('P/E ratio', f'{pe:.1f}' if pe else 'N/A')
        f2.metric('P/B ratio', f'{pb:.2f}' if pb else 'N/A')
        f3.metric('Revenue growth', f'{rg:.1%}' if rg else 'N/A')
        f4.metric('Profit margin', f'{pm:.1%}' if pm else 'N/A')
        f5, f6, f7, f8 = st.columns(4)

        peg = fund.get('peg_ratio')
        de = fund.get('debt_to_equity')
        tp = fund.get('target_price')
        bt = fund.get('beta')
        f5.metric('PEG ratio', f'{peg:.2f}' if peg else 'N/A')
        f6.metric('Debt/Equity', f'{de/100:.2f}' if de else 'N/A')
        f7.metric('Analyst target', f'${tp:.2f}' if tp else 'N/A')
        f8.metric('Beta', f'{bt:.2f}' if bt else 'N/A')
        t1, t2, t3 = st.columns(3)
        h52 = fund.get('52w_high')
        l52 = fund.get('52w_low')
        rpc = fund.get('52w_range_pct')
        t1.metric('52w High', f'${h52:.2f}' if h52 else 'N/A')
        t2.metric('52w Low', f'${l52:.2f}' if l52 else 'N/A')
        t3.metric('Range %ile', f'{rpc:.0%}' if rpc is not None else 'N/A')
        st.markdown('---')
        st.markdown('#### Scoring rule breakdown')
        st.caption(f'Strategy auto-inferred as {strategy}.')
        for rule_text in card['rules']:
            if '+' in rule_text[:6]:
                icon, color = 'checkmark', "#35C96E"
            elif '-' in rule_text[:6]:
                icon, color = 'red_circle', "#E65353"
            else:
                icon, color = 'white_circle', "#CBCBCB"
            rstyle = f'font-family:monospace;font-size:1rem;color:{color};padding:3px 0'
            st.markdown(f'<div style={rstyle}>{rule_text}</div>', unsafe_allow_html=True)
    
    # 1-year price chart with moving averages
        st.markdown('---')
        st.markdown('#### Price chart (1 year) with moving averages')
        try:
            hist = yf.Ticker(ticker).history(period='1y')
            if not hist.empty:
                hist['MA50'] = hist['Close'].rolling(50).mean()
                hist['MA200'] = hist['Close'].rolling(200).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name=ticker, line=dict(color=sig_color, width=2)))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'],
                    name='50-day MA', line=dict(color='#7c6ae6', width=1.2, dash='dot')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], name='200-day MA',
                    line=dict(color='#BA7517', width=1.2, dash='dash')))
                fig.update_layout(height=320, yaxis_tickprefix='$')
                st.plotly_chart(fig, width='content')
        except Exception:
            st.caption('Price chart unavailable')


with st.sidebar:
    st.title('Portfolio Risk Analyzer')
    st.caption('Fidelity Edition')
    st.divider()
    # Section 1: Portfolio upload
    st.subheader('Portfolio analysis')
    uploaded = st.file_uploader('Upload Fidelity positions CSV', type=['csv'])
    if uploaded:
        st.caption(f'File: {uploaded.name}')

    st.divider()

    run_btn = st.button('Run portfolio analysis', type='primary',
    width='stretch', disabled=not uploaded)

    st.divider()

    # Section 3: Standalone ticker analyzer
    st.subheader('Analyze any ticker')
    st.caption('No portfolio upload needed')
    ticker_input = st.text_input('Ticker symbol', placeholder='e.g. NVDA').upper().strip()
    analyze_btn = st.button('Analyze ticker', width='stretch')

if analyze_btn and ticker_input:
    st.header(f'Stock Analysis: {ticker_input}')
    st.divider()
    with st.spinner(f'Fetching data for {ticker_input}...'):
        fund = fetch_fundamentals(ticker_input)
    if fund.get('name', ticker_input) == ticker_input and not fund.get('pe_ratio'):
        st.error(f'Could not find data for ticker {ticker_input}. Check the symbol.')
    else:
    # Fetch price history for moving averages and relative strength
        with st.spinner('Loading price history...'):
            prices = fetch_price_history([ticker_input, 'SPY'])
    spy_6m = compute_spy_6m_return(prices)
    strategy = pick_strategy(ticker_input, fund, prices)
    # Build a mock position for the report card
    # (no real position data since user just typed a ticker)
    mock_position = {
        'ticker': ticker_input,
        'weight': 0.0, # not in portfolio
        'current_price': fund.get('current_price', 0),
        'avg_cost': fund.get('current_price', 0),
        'unrealized_pct': 0.0,
        'asset_type': 'Stock',
    }
    card = generate_report_card(mock_position, fund, {}, prices, spy_6m)
    # ■■ Render the report card ■■
    _render_report_card(card, fund, standalone=True)
    
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        positions = load_portfolio(tmp_path)
        positions = enrich_positions(positions)
    finally:
        os.unlink(tmp_path)
    return positions
    
@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple):
    return fetch_price_history(list(tickers) + ['SPY'], years=2)
    
if uploaded and run_btn:
    with st.spinner('Parsing Fidelity export...'):
        positions = load_data(uploaded.read())
    with st.spinner(f'Fetching prices for {len(positions)} tickers + SPY...'):
        prices = load_prices(tuple(positions['ticker'].tolist()))
    with st.spinner('Computing risk metrics...'):
        metrics = run_all_metrics(positions, prices)
    with st.spinner('Running scoring engine on all positions...'):
        report_cards = run_all_signals(positions, metrics, prices)
        tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'Risk Analysis', 'Rebalancing', 'Stock Insights'])
    with tab1:
        st.subheader('Portfolio overview')
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric('Total value', f'${metrics['total_value']:,.0f}')
        c2.metric('Annual return', f'{metrics['annualized_return']:.1%}')
        c3.metric('Volatility', f'{metrics['volatility']:.1%}')
        c4.metric('Sharpe ratio', f'{metrics['sharpe_ratio']:.2f}')
        c5.metric('Max drawdown', f'{metrics['max_drawdown']:.1%}')
        st.divider()
        c1,c2,c3 = st.columns(3)
        c1.metric('95% VaR ($)',
        f'${metrics['var_95_usd']:,.0f}', delta=f'{metrics['var_95_pct']:.2%} of portfolio', delta_color='inverse')
        c2.metric('99% VaR', f'{metrics['var_99_pct']:.2%}')
        c3.metric    ('CVaR 95%', f'{metrics['cvar_95']:.2%}')
            
        # Strategy mix donut + sector donut side by side
        st.divider()
        st.markdown('#### Strategy mix and sector allocation')
        strategies = [c['strategy'] for c in report_cards]
        strat_df = pd.Series(strategies).value_counts().reset_index()
        strat_df.columns = ['Strategy', 'Count']
        col1, col2 = st.columns(2)
        with col1:
            st.caption('By inferred strategy')
            fig = px.pie(strat_df, values='Count', names='Strategy',
            color_discrete_map=STRATEGY_COLORS, hole=0.42)
            st.plotly_chart(fig, width='stretch')
        with col2:
                st.caption('By sector')
                sb = metrics['sector_breakdown']
                fig2 = px.pie(sb, values='weight', names='sector', hole=0.42)
                st.plotly_chart(fig2, width='stretch')

        with tab2:
            st.subheader('Risk analysis')
            st.markdown('#### Drawdown over time')
            dd = metrics['drawdown_series']
            fig = go.Figure(go.Scatter(
            x=dd.index, y=dd.values, fill='tozeroy',
            fillcolor='rgba(216,90,48,0.15)',
            line=dict(color='#D85A30', width=1.5)))
            fig.update_layout(yaxis=dict(tickformat='.0%'), height=260)
            st.plotly_chart(fig, width='stretch')
            st.divider()
            st.markdown('#### Correlation matrix')
            corr = metrics['corr_matrix']
            fig2 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(),
            y=corr.index.tolist(), colorscale='RdBu_r', zmid=0,
            text=corr.round(2).values, texttemplate='%{text}',
            textfont=dict(size=9)))
            fig2.update_layout(height=520)
            st.plotly_chart(fig2, width='stretch')
            if metrics['beta']:
                b = metrics['beta']
                c1,c2,c3 = st.columns(3)
                c1.metric('Beta vs S&P 500', f'{b['beta']:.3f}')
                c2.metric('Alpha (annualized)', f'{b['alpha']:.2%}')
                c3.metric('R-squared', f'{b['r_squared']:.3f}')

        with tab3:
            st.subheader('Rebalancing suggestions')
            flags = metrics['rebalancing_flags']
            if not flags:
                st.success('No positions exceed the 20% concentration threshold.')
            else:
                for f in flags:
                    with st.container(border=True):
                        st.markdown(f'**{f['ticker']}** is {f['current_weight']:.1%}')
                        st.markdown(f['message'])
                c1,c2,c3 = st.columns(3)
                c1.metric('Current weight', f'{f['current_weight']:.1%}')
                c2.metric('Suggested weight', f'{f['suggested_weight']:.1%}')
                c3.metric('Vol reduction', f'{f['vol_reduction']:.2%}')

        with tab4:
            st.subheader('Stock insights')
            st.caption('Rule-based signals from your own scoring engine. '
            'Every signal is explainable — you wrote the rule.')
            buys = [c for c in report_cards if c['signal'] == 'BUY']
            holds = [c for c in report_cards if c['signal'] == 'HOLD']
            sells = [c for c in report_cards if c['signal'] == 'SELL']
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric('BUY signals', len(buys))
            sc2.metric('HOLD signals', len(holds))
            sc3.metric('SELL signals', len(sells))
            st.divider()

            # SELL first (most urgent), then HOLD, then BUY
            sorted_cards = sells + holds + buys
            for card in sorted_cards:
                label = (f'{card['ticker']} | {card['strategy']} | '
                    f'{card['signal']} | '
                    f'Score {card['total_score']}/{card['max_score']}'
                    )
                with st.expander(label, expanded=(card['signal'] == 'SELL')):
                    _render_report_card(card, card['fundamentals'])


def testing(): 
    filename = 'data/Portfolio_Positions_' + 'Jun-25-2026' + '.csv'

    positions = load_portfolio(filename) 
    print(positions[['ticker', 'shares', 'current_value', 'weight', 'asset_type']])

    # Enrich with sectors 
    positions = enrich_positions(positions) 
    print(positions[['ticker', 'sector']].to_string())

    # Fetch price history 
    tickers = positions['ticker'].tolist() 
    prices = fetch_price_history(tickers) 
    print(f'Price history shape: {prices.shape}') 
    print(prices.tail(3))

    whatever = run_all_metrics(positions, prices)

    # print(whatever)


    print('++++++++++++++++ PHASE 3 ++++++++++++++++')
    # Phase 3 test 
    tickers = positions['ticker'].tolist() + ['SPY'] # include SPY for comparison
    metrics = run_all_metrics(positions, prices)
    cards = run_all_signals(positions, metrics, prices)
    for card in cards:
        print(f"\n{'='*50}")
        print(f"{card['ticker']} | {card['strategy']} | {card['signal']}")
        print(f"Score: {card['total_score']}/{card['max_score']}")
        print('Rules:')
        for rule in card['rules']:
            print(f' {rule}')
