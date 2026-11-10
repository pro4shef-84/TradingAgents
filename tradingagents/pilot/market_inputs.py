"""Bounded supplemental observations. No brokerage writes or implicit feed fallback.

Availability for historical API data is a conservative bar-end/tick-time proxy,
not a claim to have archived the original publication or correction time.
"""
import csv
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from tradingagents.pilot.providers import alpaca_configured, alpaca_get


def timestamp(value):
    try:
        t = pd.Timestamp(value)
        if pd.isna(t) or t.tzinfo is None:
            raise ValueError()
        return t.tz_convert('UTC')
    except (TypeError, ValueError):
        raise ValueError('Expected timezone-aware timestamp') from None


def _number(value, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError('Invalid numeric market observation') from None
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError('Invalid numeric market observation')
    return result


def _fetch(kind, tickers, start, end, feed='sip', max_pages=10, page_limit=1000):
    start, end = timestamp(start), timestamp(end)
    if (not isinstance(tickers, (list, tuple)) or not 1 <= len(tickers) <= 20
            or any(not isinstance(t, str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', t) for t in tickers)):
        raise ValueError('Expected one to twenty valid symbols')
    if end <= start or end-start > pd.Timedelta(days=7):
        raise ValueError('Observation window must be positive and at most seven days')
    if feed not in ('sip', 'iex'):
        raise ValueError('Unsupported explicit feed')
    if not 1 <= max_pages <= 100 or not 1 <= page_limit <= 10000:
        raise ValueError('Invalid pagination budget')
    retrieved = pd.Timestamp.now(tz='UTC').isoformat()
    mapping = {t.replace('-', '.'): t for t in tickers}
    params = dict(symbols=','.join(mapping),start=start.isoformat(),end=end.isoformat(),
                  feed=feed,limit=page_limit,sort='asc')
    if kind == 'bars': params.update(timeframe='1Min',adjustment='raw')
    records = {t: [] for t in tickers}; seen = {t:set() for t in tickers}
    duplicates = {t:0 for t in tickers}; tokens=set(); truncated=False
    bar_timestamps={t:set() for t in tickers}
    for _ in range(max_pages):
        page=alpaca_get('/v2/stocks/'+kind,params)
        if not isinstance(page,dict) or not isinstance(page.get(kind),dict):
            raise ValueError('Malformed Alpaca response')
        for symbol, rows in page[kind].items():
            if symbol not in mapping or not isinstance(rows,list):
                raise ValueError('Unexpected symbol or malformed rows')
            symbol=mapping[symbol]
            for raw in rows:
                t=timestamp(raw.get('t'))
                if t < start or t > end: raise ValueError('Observation outside requested timestamp window')
                if kind=='bars' and t+pd.Timedelta(minutes=1)>end: continue
                identity=json.dumps(raw,sort_keys=True)
                if identity in seen[symbol]: duplicates[symbol]+=1; continue
                if kind=='bars' and t in bar_timestamps[symbol]:raise ValueError('Conflicting duplicate bar timestamp')
                bar_timestamps[symbol].add(t)
                seen[symbol].add(identity)
                record=dict(symbol=symbol,event_at=t.isoformat(),timestamp=t.isoformat(),
                    available_at=(t+pd.Timedelta(minutes=1) if kind=='bars' else t).isoformat(),
                    session=str(t.tz_convert('America/New_York').date()),provider='Alpaca',feed=feed,
                    retrieved_at=retrieved,adjustment='raw')
                if kind=='bars':
                    for field,key in [('open','o'),('high','h'),('low','l'),('close','c'),('volume','v')]:
                        record[field]=_number(raw.get(key),positive=key!='v')
                    if not record['low']<=min(record['open'],record['close'])<=max(record['open'],record['close'])<=record['high']:
                        raise ValueError('Malformed OHLC range')
                    record.update(interval_minutes=1,vwap=_number(raw['vw'],True) if raw.get('vw') is not None else None)
                elif kind=='trades':
                    record.update(price=_number(raw.get('p'),True),size=_number(raw.get('s'),True),
                        conditions=raw.get('c',[]),trade_id=raw.get('i'),exchange=raw.get('x'),
                        correction=raw.get('correction',False))
                    if not isinstance(record['conditions'],list):raise ValueError('Malformed trade conditions')
                else:
                    record.update(bid=_number(raw.get('bp')),ask=_number(raw.get('ap')),
                        bid_size=_number(raw.get('bs')),ask_size=_number(raw.get('as')),
                        conditions=raw.get('c',[]))
                    if record['ask']<record['bid']:raise ValueError('Crossed quote')
                records[symbol].append(record)
        token=page.get('next_page_token')
        if not token: break
        if not isinstance(token,str) or token in tokens:raise ValueError('Repeated or malformed Alpaca pagination token')
        tokens.add(token);params['page_token']=token
    else: truncated=True
    coverage={}
    for symbol,rows in records.items():
        rows.sort(key=lambda r:r['event_at'])
        missing=[]
        if kind=='bars':
            # Clock intervals in the requested window, including extended hours.
            # Zero-trade minutes may be absent; absence is never silently filled.
            expected=pd.date_range(start.ceil('min'),end-pd.Timedelta(minutes=1),freq='min')
            observed={r['event_at'] for r in rows}
            missing=[t.isoformat() for t in expected if t.isoformat() not in observed]
        complete=kind=='bars' and not missing and not truncated and bool(rows)
        coverage[symbol]=dict(status='complete' if complete else 'partial' if kind=='bars' or truncated else 'observed',
            complete=complete,truncated=truncated,record_count=len(rows),duplicates_removed=duplicates[symbol],
            missing_intervals=missing,start=start.isoformat(),end=end.isoformat(),
            first_at=rows[0]['event_at'] if rows else None,last_at=rows[-1]['event_at'] if rows else None,
            availability_policy='bar_end_or_tick_time_proxy; revisions unavailable',
            limitation='Historical ticks do not certify correction-free or complete tape coverage')
    return dict(provider='Alpaca',feed=feed,adjustment='raw',retrieved_at=retrieved,
                start=start.isoformat(),end=end.isoformat(),records=records,coverage=coverage,
                budget=dict(max_symbols=20,max_pages=max_pages,page_limit=page_limit,max_records=max_pages*page_limit))


def fetch_intraday(tickers,start,end,feed='sip',**kwargs):
    return _fetch('bars',tickers,start,end,feed,**kwargs)


def fetch_trades(tickers,start,end,feed='sip',**kwargs):
    return _fetch('trades',tickers,start,end,feed,**kwargs)


def fetch_quotes(tickers,start,end,feed='sip',**kwargs):
    return _fetch('quotes',tickers,start,end,feed,**kwargs)


def import_records(source,kind,cutoff=None):
    """Validate sourced JSON/CSV records; imports never imply live connectivity.

    Book updates must contain full resulting bids/asks, not deltas. A gap makes
    state invalid until an explicit full snapshot resynchronizes it.
    """
    if kind not in ('books','events','social'):raise ValueError('Unsupported import kind')
    if isinstance(source,(str,Path)):
        path=Path(source)
        if path.stat().st_size>10_000_000:raise ValueError('Import exceeds 10MB budget')
        if path.suffix.lower()=='.csv':
            with path.open(newline='') as f: source=list(csv.DictReader(f))
        else:source=json.loads(path.read_text())
    if not isinstance(source,list) or len(source)>10000:raise ValueError('Import must be a bounded record list')
    cutoff=timestamp(cutoff) if cutoff is not None else pd.Timestamp.now(tz='UTC')
    records={};states={};retrieved=pd.Timestamp.now(tz='UTC').isoformat()
    for raw in source:
        r=dict(raw);symbol=r.get('symbol')
        if not isinstance(symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}',symbol):raise ValueError('Invalid symbol')
        event=timestamp(r.get('event_at',r.get('event_time',r.get('timestamp'))))
        available=timestamp(r.get('available_at'))
        if available<event:raise ValueError('Availability precedes event timestamp')
        updated=timestamp(r['updated_at']) if r.get('updated_at') else available
        available=max(available,updated)
        url=urlsplit(str(r.get('source_url','')))
        if url.scheme!='https' or not url.hostname or url.username:raise ValueError('Expected sourced HTTPS import')
        if not isinstance(r.get('provider'),str) or not r['provider'].strip():raise ValueError('Expected provider')
        if available>cutoff or event>cutoff:continue
        r.update(event_at=event.isoformat(),event_time=event.isoformat(),timestamp=event.isoformat(),
            available_at=available.isoformat(),retrieved_at=retrieved,
            session=str(event.tz_convert('America/New_York').date()),adjustment='raw',feed='sourced_import')
        if kind=='events':
            verified=r.get('verified')
            if isinstance(verified,str) and verified in ('true','false'):
                verified=verified=='true'
            if not isinstance(verified,bool):raise ValueError('Explicit boolean event verification required')
            r['verified']=verified
        if kind=='books':
            seq=r.get('sequence')
            if isinstance(seq,str) and seq.isdigit():seq=int(seq)
            if not isinstance(seq,int) or isinstance(seq,bool) or seq<0:raise ValueError('Invalid book sequence')
            snapshot=r.get('snapshot') is True or r.get('type')=='snapshot'
            previous=states.get(symbol)
            if previous and event<previous[2]:raise ValueError('Book timestamps must be ordered')
            valid=snapshot or bool(previous and previous[1] and seq==previous[0]+1)
            for side in ('bids','asks'):
                if isinstance(r.get(side),str):r[side]=json.loads(r[side])
                levels=r.get(side)
                if not isinstance(levels,list) or not levels:raise ValueError('Expected full book state')
                for level in levels:
                    if not isinstance(level,(list,tuple)) or len(level)!=2:raise ValueError('Invalid book level')
                    _number(level[0],True);_number(level[1])
            if max(float(x[0]) for x in r['bids'])>=min(float(x[0]) for x in r['asks']):raise ValueError('Crossed book')
            r.update(valid=valid,sequence=seq,type='snapshot' if snapshot else 'update')
            states[symbol]=(seq,valid,event)
        records.setdefault(symbol,[]).append(r)
    for rows in records.values():rows.sort(key=lambda r:r['event_at'])
    return dict(provider='sourced_import',feed='sourced_import',adjustment='raw',retrieved_at=retrieved,records=records,
        coverage={s:dict(status='imported',complete=False,record_count=len(rows),
            limitation='Importer validates provenance fields; independent source completeness unverified') for s,rows in records.items()})


def connection_status(observations=None,now=None,max_age_seconds=3600):
    """Configuration is not a verified connection. Optional capability observations expire."""
    result={'alpaca':dict(status='configured' if alpaca_configured() else 'not_connected')}
    for name in ('level2','x','stocktwits'):
        result[name]=dict(status='not_connected',detail='Validated sourced import available; live provider access not verified')
    for name,observation in (observations or {}).items():
        if name not in result:continue
        status=observation.get('status')
        if status not in ('connected','insufficient_entitlement','failed','not_connected'):continue
        if status=='connected' and not observation.get('checked_at'):continue
        result[name]={'status':status}
        if observation.get('checked_at'):
            checked=timestamp(observation['checked_at']);result[name]['checked_at']=checked.isoformat()
            current=timestamp(now) if now else pd.Timestamp.now(tz='UTC')
            if checked>current:result[name]['status']='failed'
            elif status=='connected' and current-checked>pd.Timedelta(seconds=max_age_seconds):
                result[name]['status']='stale'
        for field in ('scope','feed'):
            if isinstance(observation.get(field),str):result[name][field]=observation[field][:512]
        budget=observation.get('budget')
        if isinstance(budget,dict):
            result[name]['budget']={k:v for k,v in budget.items() if k in
                ('symbols','max_symbols','max_pages','max_pages_per_endpoint','page_limit','max_records') and isinstance(v,int) and not isinstance(v,bool) and 0<=v<=1_000_000}
        capabilities=observation.get('capabilities')
        if isinstance(capabilities,dict):
            safe={}
            for capability,metadata in capabilities.items():
                if capability not in ('bars','trades','quotes','previous_close') or not isinstance(metadata,dict):continue
                safe[capability]={}
                for field,value in metadata.items():
                    if field=='status' and value in ('connected','insufficient_entitlement','failed','not_connected','partial','observed'):
                        safe[capability][field]=value
                    elif field=='truncated' and isinstance(value,bool):safe[capability][field]=value
                    elif field=='record_count' and isinstance(value,int) and value>=0:safe[capability][field]=value
            result[name]['capabilities']=safe
    return result


def for_symbol(batch,symbol):
    """Project batch acquisition into the evaluator's per-symbol input envelope."""
    rows=batch.get('records',{}).get(symbol,[])
    return {k:batch.get(k) for k in ('provider','feed','retrieved_at','adjustment','start','end')} | dict(
        symbol=symbol,session=rows[0]['session'] if rows else None,records=rows,
        coverage=batch.get('coverage',{}).get(symbol,dict(complete=False,status='missing')))


def fetch_previous_closes(tickers,session,feed='sip',max_pages=3):
    """Fetch the previous XNYS session's raw close for comparable opening gaps.

    Budget exhaustion rejects partial results. A missing symbol stays absent so
    downstream code abstains rather than substituting an adjusted close.
    """
    import exchange_calendars as xcals
    if (not isinstance(tickers,(list,tuple)) or not 1<=len(tickers)<=20 or
            any(not isinstance(t,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}',t) for t in tickers)):
        raise ValueError('Expected one to twenty valid symbols')
    if feed not in ('sip','iex') or not isinstance(max_pages,int) or not 1<=max_pages<=10:
        raise ValueError('Invalid previous-close feed or pagination budget')
    date=pd.Timestamp(session)
    if date.tzinfo is not None or date!=date.normalize():raise ValueError('Expected XNYS session date')
    calendar=xcals.get_calendar('XNYS')
    if not calendar.is_session(date):raise ValueError('Expected XNYS session date')
    previous=calendar.previous_session(date)
    previous_date=str(previous.date())
    local=pd.Timestamp(previous_date,tz='America/New_York')
    mapping={t.replace('-','.'):t for t in tickers}
    params=dict(symbols=','.join(mapping),timeframe='1Day',adjustment='raw',feed=feed,
        start=local.isoformat(),end=(local+pd.Timedelta(hours=23,minutes=59)).isoformat(),limit=1000,sort='asc')
    result={};tokens=set();retrieved=pd.Timestamp.now(tz='UTC').isoformat()
    for _ in range(max_pages):
        page=alpaca_get('/v2/stocks/bars',params)
        if not isinstance(page,dict) or not isinstance(page.get('bars'),dict):raise ValueError('Malformed previous-close response')
        for provider_symbol,rows in page['bars'].items():
            if provider_symbol not in mapping or not isinstance(rows,list):raise ValueError('Unexpected previous-close symbol')
            symbol=mapping[provider_symbol]
            for row in rows:
                observed=timestamp(row.get('t'))
                if str(observed.tz_convert('America/New_York').date())!=previous_date:
                    raise ValueError('Previous close returned stale or unexpected session')
                reference=dict(price=_number(row.get('c'),True),asof=previous_date,adjustment='raw',
                    provider='Alpaca',feed=feed,retrieved_at=retrieved,
                    available_at=calendar.session_close(previous).isoformat())
                if symbol in result and result[symbol]!=reference:raise ValueError('Conflicting previous close')
                result[symbol]=reference
        token=page.get('next_page_token')
        if not token:break
        if not isinstance(token,str) or token in tokens:raise ValueError('Repeated previous-close pagination token')
        tokens.add(token);params['page_token']=token
    else:raise ValueError('Previous-close pagination budget exhausted')
    return result
