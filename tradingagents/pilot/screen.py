"""Pre-research eligibility and interpretable queues, independent of trade plans."""
from datetime import datetime, timezone
import re
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from tradingagents.technical_scan import validate_bars
from tradingagents.technique_panel import evaluate_stock, judge, research_sort_key

POLICY = {'version':'pilot_v1', 'preferred_dollar_volume':20_000_000,
          'minimum_dollar_volume':5_000_000, 'liquidity_sessions':20,
          'preferred_spread_pct':.20, 'minimum_quote_samples':3,
          'quote_max_age_seconds':120, 'quote_window_seconds':900}
EXCLUDED = re.compile(r'\b(preferred|preference|warrants?|rights|units?|notes|debentures|bonds|ETF|ETN|fund|certificates)\b',re.I)


def utcnow(): return datetime.now(timezone.utc).isoformat()


def market_clock(now=None):
    now=pd.Timestamp(now or utcnow())
    if now.tzinfo is None: raise ValueError('Clock must include timezone')
    now=now.tz_convert('UTC')
    cal=xcals.get_calendar('XNYS')
    schedule=cal.schedule.loc[str((now-pd.Timedelta(days=14)).date()):str((now+pd.Timedelta(days=7)).date())]
    completed=schedule[schedule.close+pd.Timedelta(minutes=15)<=now]
    if completed.empty: raise ValueError('Cannot resolve completed session')
    return dict(asof=str(completed.index[-1].date()),now=now.isoformat(),
                regular_session_open=bool(((schedule.open<=now)&(now<schedule.close)).any()),
                next_close=str(schedule.loc[schedule.close>now,'close'].iloc[0]),calendar='XNYS',close_buffer_minutes=15)


def spread_assessment(samples,now=None):
    now=pd.Timestamp(now or utcnow())
    if now.tzinfo is None: raise ValueError('Timezone required')
    valid=[];seen=set(); latest=None
    for q in samples:
        try:
            bid,ask=float(q['bid']),float(q['ask'])
            if not np.isfinite([bid,ask]).all() or not 0<bid<=ask: continue
            observed=pd.Timestamp(q['observed_at']); ts=pd.Timestamp(q['quote_at'])
            if observed.tzinfo is None or ts.tzinfo is None: continue
            if not 0<=(now-observed).total_seconds()<=POLICY['quote_window_seconds']: continue
            if not 0<=(observed-ts).total_seconds()<=POLICY['quote_max_age_seconds']: continue
            if not q.get('regular_session'): continue
            # Re-fetching a cached quote does not manufacture independent samples.
            key=(q['provider'],q.get('feed'),ts.isoformat())
            if key in seen: continue
            seen.add(key);latest=max(latest,ts) if latest is not None else ts
            valid.append((ask-bid)/((ask+bid)/2)*100)
        except (ValueError,TypeError,KeyError): continue
    if latest is None or (now-latest).total_seconds()>POLICY['quote_max_age_seconds']:
        return dict(status='unchecked',median_pct=None,samples=0,reason='No fresh timestamped regular-session bid/ask')
    median=float(np.median(valid))
    status='sampling' if len(valid)<3 else 'preferred' if median<=POLICY['preferred_spread_pct'] else 'wide'
    return dict(status=status,median_pct=median,samples=len(valid),reason='Spread is a review flag, never an eligibility veto')


def identity_check(member,quote=None):
    name=member.get('name','')
    if EXCLUDED.search(name): return 'excluded_security_type'
    if quote:
        if quote.get('symbol')!=member['ticker']: return 'symbol_mismatch'
        if quote.get('quote_type') not in (None,'EQUITY'): return 'excluded_security_type'
        if quote.get('exchange') in ('PNK','OTC','OTCQB','OTCQX'): return 'excluded_exchange'
        if quote.get('quote_type')=='EQUITY' and quote.get('exchange') in ('NMS','NGM','NCM','NYQ','ASE','PCX','BTS','NYSE','NASDAQ','AMEX','ARCA','BATS'):
            return 'provider_verified_equity'
    return 'listing_screened_identity_pending'


def research_queue(assessment,context,agents,eligibility):
    if eligibility=='excluded':return 'excluded'
    if eligibility=='data_review':return 'data_review'
    direction=assessment.get('evidence_direction','neutral')
    if direction=='mixed':return 'conflicting'
    if direction=='bearish':return 'bearish'
    if direction=='bullish':return 'continuation' if context.get('regime')=='uptrend' else 'recovery'
    if any(a['status']=='watch' for a in agents):return 'developing'
    return 'no_current_signal'


def build_row(member,bars,benchmark,asof,quote=None,samples=(),evidence=None,now=None):
    base=dict(ticker=member['ticker'],name=member.get('name',''),sector=member.get('sector',''),market_cap_rank=int(member.get('rank',9999)),
              asof=asof,quote=quote,spread=spread_assessment(samples,now),identity=identity_check(member,quote),
              eligibility='data_review',liquidity_tier='unchecked',median_dollar_volume=None,research_queue='data_review',
              reasons=[],agents=[],context={},close=None,chart=[],source='Yahoo adjusted daily OHLCV',
              evidence=evidence or {},observed_at=now or utcnow())
    try:
        d=validate_bars(bars.loc[:asof])
        if d.empty or str(d.index[-1].date())!=asof: raise ValueError('Stale or missing completed-session bar')
        # Verify consecutive session coverage, not calendar-day gaps across weekends.
        expected=xcals.get_calendar('XNYS').sessions_in_range(d.index[max(0,len(d)-20)],asof)
        if not expected.isin(d.index).all(): raise ValueError('Missing recent exchange-session bars')
        base['close']=float(d.Close.iloc[-1]);base['bars']=len(d)
        if len(d)<20: raise ValueError('20 sessions needed for liquidity estimate')
        median=float((d.Close*d.Volume).tail(20).median())
        base['median_dollar_volume']=median
        base['liquidity_tier']='preferred' if median>=20e6 else 'secondary' if median>=5e6 else 'thin'
        panel=evaluate_stock(d,benchmark=benchmark,asof=asof)
        assessment=judge(panel,d)
        base.update(context=panel['context'],agents=panel['agents'],assessment=assessment)
        base['chart']=[dict(date=str(date.date()),close=float(row.Close)) for date,row in d.tail(120).iterrows()]
        if base['identity'].startswith('excluded') or base['identity']=='symbol_mismatch':
            base.update(eligibility='excluded',reasons=['Instrument requires separate universe'])
        elif median<5e6:
            base.update(eligibility='excluded',reasons=['Median daily dollar volume below $5m'])
        else:
            base['eligibility']='eligible' if base['identity']=='provider_verified_equity' else 'provisional'
            if base['eligibility']=='provisional': base['reasons'].append('Listing screened; quote-provider identity confirmation pending')
        base['research_queue']=research_queue(assessment,panel['context'],panel['agents'],base['eligibility'])
        base['signal_date']=asof
        base['signal_freshness']='Latest completed-session condition; persistent indicators are not necessarily new events'
        base['data_quality']='valid_completed_bars'
    except (ValueError,KeyError,IndexError) as exc:
        base['reasons'].append(str(exc));base['data_quality']='needs_review'
    return base


def queue_key(row):
    assessment=row.get('assessment',{})
    return ({'eligible':0,'provisional':1,'data_review':2,'excluded':3}[row['eligibility']],
            {'preferred':0,'secondary':1,'unchecked':2,'thin':3}[row['liquidity_tier']],
            *research_sort_key({**assessment,'decision':assessment.get('decision','insufficient_data'),'market_cap_rank':row['market_cap_rank']}))
