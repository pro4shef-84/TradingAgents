"""Read-only providers. Feed identity and timestamp gaps are never hidden."""
import os
import pandas as pd
import requests
import yfinance as yf
from tradingagents.pilot.screen import market_clock, utcnow


def yahoo_quote(ticker):
    info=yf.Ticker(ticker).get_info()
    observed=utcnow()
    ts=info.get('regularMarketTime')
    return dict(symbol=info.get('symbol'),provider='Yahoo Finance',feed='provider quote snapshot',
        quote_type=info.get('quoteType'),exchange=info.get('exchange'),
        bid=info.get('bid'),ask=info.get('ask'),price=info.get('regularMarketPrice'),
        price_at=pd.Timestamp(ts,unit='s',tz='UTC').isoformat() if ts else None,
        quote_at=None,observed_at=observed,regular_session=market_clock(observed)['regular_session_open'],
        note='Bid/ask timestamp not supplied; last-trade timestamp cannot certify spread freshness')


def daily_bars(tickers,asof):
    start=(pd.Timestamp(asof)-pd.Timedelta(days=550)).strftime('%Y-%m-%d')
    end=(pd.Timestamp(asof)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    raw=yf.download(tickers,start=start,end=end,auto_adjust=True,group_by='ticker',threads=4,progress=False,timeout=20)
    result={}
    for ticker in tickers:
        try:
            d=raw[ticker] if isinstance(raw.columns,pd.MultiIndex) else raw
            d=d[['Open','High','Low','Close','Volume']].dropna(how='all')
            if d.index.tz is not None: d.index=d.index.tz_localize(None)
            result[ticker]=d.loc[:asof]
        except KeyError: continue
    return result


def alpaca_headers():
    from pathlib import Path
    from dotenv import dotenv_values
    local=dotenv_values(Path(__file__).resolve().parents[2]/'.env')
    key=os.environ.get('APCA_API_KEY_ID') or local.get('APCA_API_KEY_ID')
    secret=os.environ.get('APCA_API_SECRET_KEY') or local.get('APCA_API_SECRET_KEY')
    if not key or not secret: raise ValueError('Configure APCA_API_KEY_ID and APCA_API_SECRET_KEY in local .env')
    return {'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret}


def alpaca_configured():
    try: alpaca_headers(); return True
    except ValueError: return False


class AlpacaEntitlementError(ValueError):
    """Explicit authentication/feed authorization failure, never a fallback trigger."""
    status = 'insufficient_entitlement'


def alpaca_get(path,params=None,assets=False):
    # Fixed read-only endpoints. No arbitrary URLs, order endpoints or write methods.
    allowed={'/v2/stocks/bars','/v2/stocks/trades','/v2/stocks/quotes','/v2/stocks/quotes/latest','/v2/assets','/v1beta1/news'}
    if path not in allowed:raise ValueError('Unsupported read-only endpoint')
    host='https://paper-api.alpaca.markets' if assets else 'https://data.alpaca.markets'
    for attempt in range(3):
        try:
            r=requests.get(host+path,params=params,headers=alpaca_headers(),timeout=20)
        except requests.RequestException:
            if attempt<2:
                import time
                time.sleep(1+attempt)
                continue
            raise ValueError('Alpaca network request failed after bounded retries') from None
        if r.status_code==429 or r.status_code>=500:
            if attempt<2:
                import time
                time.sleep(1+attempt)
                continue
        if r.status_code in (401,403):
            raise AlpacaEntitlementError(f'Alpaca HTTP {r.status_code}; authentication or feed entitlement unavailable. No automatic feed substitution.')
        if r.status_code!=200:
            raise ValueError(f'Alpaca HTTP {r.status_code}; verify paper keys and feed entitlement. No automatic feed substitution.')
        try:
            return r.json()
        except ValueError:
            raise ValueError('Malformed Alpaca JSON response') from None
    raise ValueError('Alpaca retry limit reached')


def alpaca_assets():
    return {a['symbol'].replace('.','-'):a for a in alpaca_get('/v2/assets',{'status':'active','asset_class':'us_equity'},assets=True)}


def alpaca_daily_bars(tickers,asof,assets=None,max_pages=100):
    if not isinstance(max_pages,int) or not 1<=max_pages<=100:raise ValueError('Invalid daily pagination budget')
    assets=assets or {}
    mapping={assets.get(t,{}).get('symbol',t):t for t in tickers}
    end=pd.Timestamp(asof,tz='America/New_York')+pd.Timedelta(hours=23,minutes=59)
    if end>pd.Timestamp.now(tz='UTC')-pd.Timedelta(minutes=16):
        end=pd.Timestamp.now(tz='UTC')-pd.Timedelta(minutes=16)
    params={'symbols':','.join(mapping),'timeframe':'1Day','feed':'sip','adjustment':'all',
        'start':(pd.Timestamp(asof)-pd.Timedelta(days=550)).strftime('%Y-%m-%d'),
        'end':end.isoformat(),'limit':10000,'sort':'asc'}
    collected={t:[] for t in tickers};tokens=set()
    for _ in range(max_pages):
        result=alpaca_get('/v2/stocks/bars',params)
        for symbol,rows in result.get('bars',{}).items():
            if symbol in mapping:collected[mapping[symbol]].extend(rows)
        token=result.get('next_page_token')
        if not token:break
        if token in tokens:raise ValueError('Repeated Alpaca pagination token')
        tokens.add(token);params={**params,'page_token':token}
    else:raise ValueError('Alpaca daily pagination budget exhausted; partial result rejected')
    output={}
    for t,rows in collected.items():
        if not rows:continue
        d=pd.DataFrame(rows)
        d.index=pd.to_datetime(d.pop('t'),utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
        d=d.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'})
        output[t]=d[['Open','High','Low','Close','Volume']].sort_index().loc[:asof]
    return output


def alpaca_quotes(tickers,assets=None):
    assets=assets or {};mapping={assets.get(t,{}).get('symbol',t):t for t in tickers}
    result=alpaca_get('/v2/stocks/quotes/latest',{'symbols':','.join(mapping),'feed':'iex'})
    observed=utcnow();regular=market_clock(observed)['regular_session_open']
    output={}
    for symbol,q in result.get('quotes',{}).items():
        if symbol not in mapping:continue
        t=mapping[symbol];a=assets.get(t,{})
        output[t]=dict(symbol=t,provider='Alpaca',feed='IEX only; not consolidated NBBO',bid=q.get('bp'),ask=q.get('ap'),
            quote_at=q.get('t'),price=None,price_at=None,observed_at=observed,regular_session=regular,
            quote_type='EQUITY' if a.get('class')=='us_equity' else None,exchange=a.get('exchange'),
            note='Single-exchange quote; not full-market spread. Daily volume uses SIP separately.')
    return output
