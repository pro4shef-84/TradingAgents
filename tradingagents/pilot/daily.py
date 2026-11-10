"""Shared price, news and technique acquisition for trade automation."""
import json
from pathlib import Path

import pandas as pd

from tradingagents.pilot import providers
from tradingagents.pilot.screen import build_row, queue_key, research_queue
from tradingagents.pilot.store import Store


def save_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def source_directory(root):
    choices=sorted((Path(root)/'results').glob('top*/universe.csv'),reverse=True)
    if not choices: raise ValueError('No universe.csv available; supply a source universe first')
    return choices[0].parent


def _read_bars(path):
    frame=pd.read_csv(path,index_col=0,parse_dates=True)
    frame.index=pd.to_datetime(frame.index).tz_localize(None)
    return frame


def _load_rows(root, directory, asof, now, refresh, state):
    source=source_directory(root)
    universe=pd.read_csv(source/'universe.csv').fillna('').sort_values('rank').head(2000)
    members=universe.to_dict('records')
    store=Store(Path(root)/'results/pilot/pilot.sqlite3')
    old={r['ticker']:r for r in store.latest()}
    prices=directory/'prices';prices.mkdir(exist_ok=True)
    errors={};quotes={};assets={}
    state_path=directory/'state.json'
    tickers=[m['ticker'] for m in members]
    if refresh:
        assets=providers.alpaca_assets()
        for offset in range(0,len(tickers)+1,50):
            batch=(tickers+['SPY'])[offset:offset+50]
            pending=[t for t in batch if not (prices/f'{t}.csv').exists()]
            if not pending: continue
            try:
                data=providers.alpaca_daily_bars(pending,asof,assets)
                for ticker in pending:
                    if ticker not in data or data[ticker].empty:
                        errors[ticker]='No returned completed daily bars';continue
                    data[ticker].to_csv(prices/f'{ticker}.csv',index_label='Date')
                quotes.update(providers.alpaca_quotes(pending,assets))
            except Exception as exc:
                for ticker in pending: errors[ticker]=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
            state.update(stage='prices',processed=min(offset+len(batch),len(tickers)+1),errors=errors)
            save_json(state_path,state)
    manifest_choices=sorted((Path(root)/'results/pilot_runs').glob('*/manifest.json'),reverse=True)
    cached_dir=manifest_choices[0].parent if manifest_choices else source/'prices'
    def locate(ticker):
        for path in (prices/f'{ticker}.csv',cached_dir/f'{ticker}.csv',source/'prices'/f'{ticker}.csv',source/f'{ticker}_benchmark.csv'):
            if path.exists(): return path
        return None
    benchmark_path=locate('SPY')
    benchmark=_read_bars(benchmark_path) if benchmark_path else None
    rows=[]
    for member in members:
        ticker=member['ticker'];path=locate(ticker)
        if path is None:
            errors.setdefault(ticker,'Missing local price history');continue
        # Failed refresh must never relabel old cached data as newly fetched Alpaca data.
        fresh=(prices/f'{ticker}.csv').exists()
        if refresh and not fresh: continue
        quote=quotes.get(ticker) or old.get(ticker,{}).get('quote')
        try:
            bars=_read_bars(path)
            row=build_row(member,bars,benchmark,asof,quote=quote,
                          samples=store.samples(ticker,'Alpaca'),now=now)
            row['source']='Alpaca adjusted SIP daily OHLCV' if fresh else old.get(ticker,{}).get('source','Cached daily bars; provider unverified')
            row['price_file']=str(path.relative_to(root))
            rows.append(row)
            store.save(row)
        except Exception as exc:
            errors[ticker]=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
    return rows,errors,dict(universe_file=str(source/'universe.csv'),universe_count=len(members),
                            price_mode='refreshed' if refresh else 'local_cache',
                            successful_symbols=len(rows),failed_symbols=len(errors),
                            benchmark_available=benchmark is not None,
                            benchmark_fresh=bool(benchmark is not None and not benchmark.empty
                                and (prices/'SPY.csv').exists() and str(benchmark.index[-1].date())==asof))


def _news(root, directory, now, refresh):
    path=directory/'news.json'
    if path.exists(): return json.loads(path.read_text())
    if not refresh:
        saved=Path(root)/'results/pilot/chatter.json'
        return json.loads(saved.read_text()) if saved.exists() else dict(items=[],sources={'alpaca_news':{'status':'not_collected'}})
    from tradingagents.pilot.morning import collect, top_tickers
    tickers=top_tickers(source_directory(root))
    items,status=collect(tickers,now,(pd.Timestamp(now)-pd.Timedelta(hours=24)).isoformat())
    from tradingagents.pilot.chatter import discord
    try:
        social,ds=discord(tickers,now)
    except Exception as exc:
        social=[];ds={'status':'error','error':type(exc).__name__}
    data=dict(items=items+social,sources={'alpaca_news':status,'discord':ds})
    if status['status']=='connected':save_json(path,data)
    return data


def _enrich(root, rows, now, asof, directory):
    from tradingagents.pilot.market_inputs import fetch_intraday, fetch_trades, fetch_quotes, for_symbol
    from tradingagents.technique_panel import evaluate_stock, judge
    candidates=sorted([r for r in rows if r.get('eligibility') in ('eligible','provisional') and
                       r.get('research_queue') in ('continuation','recovery','bearish','conflicting','developing')],key=queue_key)[:20]
    if not candidates: return {'status':'no_candidates'}
    symbols=[r['ticker'] for r in candidates]
    import exchange_calendars as xcals
    calendar=xcals.get_calendar('XNYS')
    session=pd.Timestamp(asof)
    end=calendar.session_close(session).isoformat()
    start=calendar.session_open(session).isoformat()
    inputs={};failures={}
    for key,fetch in [('intraday',fetch_intraday),('trades',fetch_trades),('quotes',fetch_quotes)]:
        try:
            if key=='intraday':
                current=fetch(symbols,start,end)
                previous=calendar.previous_session(session)
                older=fetch(symbols,calendar.session_open(previous).isoformat(),calendar.session_close(previous).isoformat())
                for symbol in symbols:
                    current['records'][symbol]=older['records'].get(symbol,[])+current['records'].get(symbol,[])
                    before=older['coverage'][symbol];after=current['coverage'][symbol]
                    after.update(complete=before['complete'] and after['complete'],
                                 missing_intervals=before.get('missing_intervals',[])+after.get('missing_intervals',[]),
                                 prior_session=before,latest_session=dict(after))
                inputs[key]=current
            else:
                # Fair per-symbol bounded samples prevent the first active ticker consuming the batch budget.
                combined=None
                sample_start=(pd.Timestamp(end)-pd.Timedelta(minutes=5)).isoformat()
                for symbol in symbols:
                    try:
                        sample=fetch([symbol],sample_start,end,max_pages=1,page_limit=1000)
                    except Exception as exc:
                        failures.setdefault(key,{})[symbol]=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
                        continue
                    if combined is None:combined=sample
                    else:
                        combined['records'].update(sample['records']);combined['coverage'].update(sample['coverage'])
                if combined is not None:inputs[key]=combined
        except Exception as exc: failures[key]=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
    save_json(directory/'supplemental.json',inputs)
    imports={}
    for kind in ('events','books','pairs'):
        path=Path(root)/'results/pilot/imports'/f'{kind}.json'
        if path.exists():imports[kind]=json.loads(path.read_text())
    # Acquisition contract is finalized alongside the evaluator; normalize at one boundary below.
    for r in candidates:
        supplemental={'cutoff':now,'symbol':r['ticker']}
        for kind,result in inputs.items():
            dataset=for_symbol(result,r['ticker'])
            if isinstance(dataset,dict): supplemental[kind]=dataset
        for kind,batch in imports.items():
            if kind=='pairs':
                if r['ticker'] in batch:supplemental['pairs']=batch[r['ticker']]
            else:supplemental['book' if kind=='books' else kind]=for_symbol(batch,r['ticker'])
        if not any(k in supplemental for k in ('intraday','trades','quotes')):continue
        path=Path(root)/r['price_file'];bars=_read_bars(path);bars.attrs['adjustment']='all'
        panel=evaluate_stock(bars,asof=asof,supplemental=supplemental)
        # Retain SPY-dependent daily specialists already evaluated against the benchmark.
        prior={a['id']:a for a in r['agents']}
        for agent in panel['agents']:
            if agent['id'] in (48,): agent.update(prior[agent['id']])
        assessment=judge(panel,bars)
        r.update(agents=panel['agents'],assessment=assessment,
                 research_queue=research_queue(assessment,r.get('context',{}),panel['agents'],r['eligibility']))
    return dict(status='partial' if failures else 'collected',symbols=len(symbols),errors=failures)
