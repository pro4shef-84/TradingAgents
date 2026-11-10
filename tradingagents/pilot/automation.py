"""Local research scheduler and durable, at-most-once Slack observation alerts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

import exchange_calendars as xcals
import pandas as pd
import requests

from tradingagents.pilot.daily import _load_rows,_news,_enrich,save_json
from tradingagents.pilot.chatter import news as fetch_news
from tradingagents.pilot.market_inputs import fetch_intraday
from tradingagents.pilot.operations import slack_webhook
from tradingagents.pilot.screen import market_clock,utcnow,queue_key
from tradingagents.pilot.signal_monitor import advance,new_state,TERMINAL
from tradingagents.pilot.trade_planner import build_plans

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/automation'
CONFIG=BASE/'config.json'
STATE=BASE/'state.json'
LABEL='com.shef.market-trading.trade-alerts'
MATERIAL_NEWS=re.compile(r'bankrupt|going.concern|trading.halt|restatement|fraud|investigation|sued|lawsuit|clinical.*fail|FDA.*reject|withdraw.*guidance|convertible.*offering',re.I)


def needs_review(plan,item,baseline):
    ticker=plan['ticker']
    core=next((w for w in plan.get('name','').split() if len(w)>=4 and w.lower() not in ('class','common','stock','company')),'')
    return (ticker in item['symbols'] and MATERIAL_NEWS.search(item['title'])
            and (ticker.lower() in item['title'].lower() or core and core.lower() in item['title'].lower())
            and pd.Timestamp(item.get('updated_at') or item['published_at'])>pd.Timestamp(baseline))


def load(path,default):
    return json.loads(path.read_text()) if path.exists() else default


def key(plan):return plan['trade_session']+':'+plan['ticker']


def enqueue(state,event_id,text):
    state.setdefault('outbox',{}).setdefault(event_id,dict(status='pending',text=text))


def flush(state,config,send=True):
    if not send:return
    webhook=slack_webhook()
    if not webhook:raise ValueError('Slack webhook is not configured')
    if hashlib.sha256(webhook.encode()).hexdigest()!=config['destination_sha256']:
        raise ValueError('Slack destination changed; posting suspended pending configuration review')
    for identifier,item in state.setdefault('outbox',{}).items():
        if item['status']!='pending':continue
        item.update(status='sending',attempted_at=utcnow());save_json(STATE,state)
        try:
            response=requests.post(webhook,json={'text':item['text'],'unfurl_links':False,'unfurl_media':False},timeout=15)
            item.update(status='sent' if response.status_code==200 and response.text.strip()=='ok' else 'rejected',http_status=response.status_code)
        except requests.RequestException:item['status']='uncertain'
        item['completed_at']=utcnow();save_json(STATE,state)


def card_text(plan):
    icon='🟢' if plan['side']=='LONG' else '🔴'
    techniques=', '.join(f'#{a["id"]} {a["name"]}' for a in plan.get('matching_daily_techniques',[]))
    return (f'{icon} *{plan["ticker"]} | {plan["side"]} near ${plan["entry"]:.2f}*\n'
        f'Trigger: {plan.get("trigger", "confirmation and retest")}\n'
        f'SL: ${plan["stop"]:.2f} (−{plan["stop_loss_pct"]:.2f}%)\n'
        f'TP1: ${plan["targets"][0]:.2f} (+{plan["target_profit_pct"][0]:.2f}%) — exit 50%\n'
        f'TP2: ${plan["targets"][1]:.2f} (+{plan["target_profit_pct"][1]:.2f}%) — exit remainder\n'
        f'Evidence: {techniques}')


def morning(state,config,session,now):
    directory=BASE/'research'/session;directory.mkdir(parents=True,exist_ok=True)
    progress=load(directory/'state.json',dict(started_at=now.isoformat(),asof=market_clock(now)['asof']))
    rows_path=directory/'rows.json'
    if rows_path.exists():rows=load(rows_path,[]);coverage=load(directory/'coverage.json',{})
    else:
        rows,errors,coverage=_load_rows(ROOT,directory,progress['asof'],progress['started_at'],True,progress)
        if errors:raise ValueError(f'Daily acquisition incomplete: {len(errors)} symbols; retrying later')
        save_json(rows_path,rows);save_json(directory/'coverage.json',coverage)
    news=_news(ROOT,directory,now.isoformat(),True)
    if news.get('sources',{}).get('alpaca_news',{}).get('status')!='connected':
        raise ValueError('Morning news collection incomplete; retrying later')
    supplemental={}
    for side,queues in [('long',('continuation','recovery')),('short',('bearish',))]:
        members=sorted([r for r in rows if r['research_queue'] in queues and r['eligibility'] in ('eligible','provisional')],key=queue_key)[:10]
        side_dir=directory/side;side_dir.mkdir(exist_ok=True)
        try:supplemental[side]=_enrich(ROOT,members,now.isoformat(),progress['asof'],side_dir)
        except Exception as exc:supplemental[side]=dict(status='partial',error=type(exc).__name__)
    active=[s['plan']['ticker'] for s in state.get('signals',{}).values() if s['status']=='active']
    result=build_plans(ROOT,rows,session,exclude=active)
    # Existing active observations carry across days; pending lists are replaced.
    for s in state.setdefault('signals',{}).values():
        if s['plan']['trade_session']==session and s['status'] in ('pending','armed'):
            s['status']='expired'
    symbols=[p['ticker'] for p in result['plans']]
    if symbols:
        try:
            selected_status=[]
            for offset in range(0,len(symbols),5):
                extra,status=fetch_news(symbols[offset:offset+5],now.isoformat(),(now-pd.Timedelta(days=3)).isoformat())
                if status.get('truncated'):raise ValueError('Selected-stock news coverage capped')
                news['items']=list({i['id']:i for i in news['items']+extra}.values());selected_status.append(status)
            news['sources']['selected_news']=selected_status
        except Exception as exc:raise ValueError('Selected-stock news refresh incomplete; plans withheld') from None
    # Material new headlines are a review gate, not an inferred bull/bear vote.
    reviewed=config.get('reviewed_plans',{})
    retained=[]
    for plan in result['plans']:
        ticker=plan['ticker']
        baseline=pd.Timestamp(reviewed.get(ticker,pd.Timestamp(progress['asof']+' 16:00',tz='America/New_York').isoformat()))
        flagged=[n for n in news['items'] if needs_review(plan,n,baseline)]
        if flagged:
            result['rejected'][ticker]='new_material_news_requires_review'
            continue
        plan['news']=[dict(title=n['title'],url=n['url'],published_at=n['published_at'])
                      for n in news['items'] if ticker in n['symbols']][:3]
        retained.append(plan);state['signals'][key(plan)]=new_state(plan)
    result['plans']=retained
    result.update(prepared_at=utcnow(),trade_session=session,data_session=progress['asof'],coverage=coverage,supplemental=supplemental)
    save_json(directory/'plans.json',result);save_json(directory/'news.json',news);save_json(directory/'enriched_rows.json',rows)
    text=f'*Morning trade plans | {session} | UPDATED PENDING LIST*\nReplaces earlier untriggered ideas. Price alerts: {config["feed_label"]}; checks every minute.\n\n'
    text+='\n\n'.join(card_text(p) for p in result['plans']) or 'No setups meet the entry/risk rules today.'
    events=[e for e in config.get('economic_events',[]) if e['session']==session]
    if events:text+='\n\n'+ ' · '.join(e['display'] for e in events)
    text+='\n\n<https://www.bls.gov/schedule/2026/11_sched_list.htm|Economic calendar>'
    enqueue(state,session+':morning',text)
    state.setdefault('jobs',{})[session+':morning']=dict(status='complete',completed_at=utcnow(),plans=len(result['plans']))


def cpi_headline(session):
    try:
        response=requests.get('https://www.bls.gov/feed/cpi.rss',timeout=15)
        if response.status_code!=200:return 'CPI source unavailable; check BLS before using the plans.'
        tree=ET.fromstring(response.content)
        for item in tree.findall('.//item'):
            published=item.findtext('pubDate')
            if not published:continue
            dt=pd.Timestamp(parsedate_to_datetime(published)).tz_convert('America/New_York')
            if str(dt.date())!=session:continue
            title=re.sub('<[^>]+>',' ',item.findtext('title') or '')
            description=re.sub('<[^>]+>',' ',item.findtext('description') or '')
            return (title+' — '+' '.join(description.split()))[:400]
        return 'Today’s CPI release is not yet visible in the BLS feed; check the source.'
    except (requests.RequestException,ET.ParseError,ValueError,TypeError):
        return 'CPI source unavailable; check BLS before using the plans.'


def premarket(state,config,session,now):
    signals=[s for s in state.get('signals',{}).values() if s['plan']['trade_session']==session and s['status'] in ('pending','armed')]
    symbols=[s['plan']['ticker'] for s in signals]
    lines=[f'*Premarket update | {session}*']
    if any(e['session']==session and e['kind']=='CPI' for e in config.get('economic_events',[])):
        lines += [cpi_headline(session),'<https://www.bls.gov/cpi/|BLS CPI release>']
    items=[]
    if symbols:
        try:
            items,coverage=fetch_news(symbols+['SPY'],now.isoformat(),(now-pd.Timedelta(hours=6)).isoformat())
            if coverage.get('truncated'):raise ValueError('Premarket news coverage capped')
            save_json(BASE/'research'/session/'premarket_news.json',dict(items=items,coverage=coverage))
        except Exception:raise ValueError('Premarket news refresh incomplete; new entry alerts withheld') from None
        baseline=state['jobs'][session+':morning']['completed_at']
        for s in signals:
            if any(needs_review(s['plan'],item,baseline) for item in items):
                s['status']='review_required'
                s['history'].append(dict(kind='NEW_MATERIAL_NEWS',at=now.isoformat()))
                lines.append(s['plan']['ticker']+': PAUSED — new material headline requires review')
        # Current IEX observations are labeled and timestamped; no SIP substitution.
        try:
            batch=fetch_intraday(symbols,(now-pd.Timedelta(minutes=30)).isoformat(),now.isoformat(),feed=config['feed'],max_pages=4)
            save_json(BASE/'research'/session/'premarket_prices.json',batch)
            for s in signals:
                records=batch['records'].get(s['plan']['ticker'],[])
                if records:
                    last=records[-1];age=(now-pd.Timestamp(last['available_at'])).total_seconds()
                    lines.append(f'{s["plan"]["ticker"]}: last observed ${last["close"]:.2f} ({"fresh" if age<=180 else "stale"}; {config["feed"].upper()})')
        except Exception as exc:lines.append('Premarket observations unavailable: '+type(exc).__name__)
    for item in sorted(items,key=lambda n:n['published_at'],reverse=True)[:3]:
        title=item['title'].replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        lines.append('• '+title[:180])
    macro=sorted([n for n in items if re.search(r'\bCPI\b|consumer.price|inflation',n['title'],re.I)],key=lambda n:n['published_at'],reverse=True)
    if macro:
        lines.append('Reported macro headline: '+macro[0]['title'].replace('<','&lt;').replace('>','&gt;')[:220])
    lines.append('Entry confirmations start after the first regular-session 15-minute candle.')
    enqueue(state,session+':premarket','\n'.join(lines))
    state.setdefault('jobs',{})[session+':premarket']=dict(status='complete',completed_at=utcnow())


def event_text(event,plan,config,now):
    labels={'ENTRY_SIGNAL_OBSERVED':'ENTRY SIGNAL OBSERVED','TP1_LEVEL_OBSERVED':'TP1 CONDITION OBSERVED — planned exit 50%',
            'TP2_LEVEL_OBSERVED':'TP2 CONDITION OBSERVED — planned exit remainder','STOP_LEVEL_OBSERVED':'STOP CONDITION OBSERVED — planned exit remaining',
            'SETUP_EXPIRED':'SETUP EXPIRED','AMBIGUOUS_INTERVAL':'REVIEW REQUIRED — stop and target in the same minute',
            'AMBIGUOUS_ENTRY_INTERVAL':'REVIEW REQUIRED — entry interval crossed an exit level','TIME_EXIT':'TIME EXIT REFERENCE'}
    text=f'*{plan["ticker"]} | {plan["side"]} | {labels[event["kind"]]}*'
    if event.get('price') is not None:text+=f'\nObserved minute close: ${event["price"]:.2f}'
    if event.get('level') is not None:
        sign=1 if plan['side']=='LONG' else -1
        pct=sign*(event['level']/plan['entry']-1)*100
        text+=f'\nReference level: ${event["level"]:.2f} ({pct:+.2f}% from planned entry)'
    if event['kind']=='STOP_LEVEL_OBSERVED' and event.get('bar_open') is not None:
        sign=1 if plan['side']=='LONG' else -1
        if sign*(event['bar_open']-plan['stop'])<0:
            text+=f'\nMinute opened at ${event["bar_open"]:.2f}, beyond the planned stop.'
    text+=f'\nPlanned entry ${plan["entry"]:.2f} · SL ${plan["stop"]:.2f} · TP ${plan["targets"][0]:.2f} / ${plan["targets"][1]:.2f}'
    at=pd.Timestamp(event['at']).tz_convert('America/Denver')
    text+=f'\nObserved {at.strftime("%b %d %I:%M %p %Z")} · {config["feed_label"]}'
    if (now-pd.Timestamp(event['at'])).total_seconds()>180:text+=' · catch-up observation'
    return text


def monitor(state,config,session,opening,closing,now):
    for s in state.get('signals',{}).values():
        if s['status'] in ('pending','armed') and s['plan']['trade_session']<session:s['status']='expired'
    watching=[s for s in state.get('signals',{}).values() if s['status'] not in TERMINAL and s['plan']['trade_session']<=session]
    if not watching:return
    cache=state.setdefault('bar_cache',{})
    if cache.get('session')!=session:cache.clear();cache.update(session=session,records={},last_fetch=None)
    start=opening if not cache['last_fetch'] else max(opening,pd.Timestamp(cache['last_fetch'])-pd.Timedelta(minutes=2))
    end=min(closing,now-pd.Timedelta(seconds=5))
    if end<=start:return
    symbols=list(dict.fromkeys(s['plan']['ticker'] for s in watching))[:20]
    batch=fetch_intraday(symbols,start.isoformat(),end.isoformat(),feed=config['feed'],max_pages=10)
    for ticker,records in batch['records'].items():
        existing={r['event_at']:r for r in cache['records'].get(ticker,[])}
        existing.update({r['event_at']:r for r in records})
        cache['records'][ticker]=sorted(existing.values(),key=lambda r:r['event_at'])
    cache['last_fetch']=end.isoformat()
    stale=[]
    calendar=xcals.get_calendar('XNYS')
    for s in watching:
        ticker=s['plan']['ticker']
        if ticker not in symbols:continue
        if s['status'] in ('pending','armed') and s['plan']['trade_session']<session:
            s['status']='expired';continue
        records=cache['records'].get(ticker,[])
        if opening+pd.Timedelta(minutes=3)<now<closing and (not records or (now-pd.Timestamp(records[-1]['available_at'])).total_seconds()>180):stale.append(ticker)
        ready=state.get('jobs',{}).get(session+':premarket',{}).get('status')=='complete'
        events=advance(s,records,opening,closing,now,allow_entries=ready)
        if s['status']=='active' and now>=closing:
            age=len(calendar.sessions_in_range(s['plan']['trade_session'],session))
            if age>=5:
                s['status']='time_exit';event=dict(kind='TIME_EXIT',ticker=ticker,side=s['plan']['side'],at=closing.isoformat())
                if records:event['level']=records[-1]['close']
                s['history'].append(event);events.append(event)
        for event in events:
            identifier=key(s['plan'])+':'+event['kind']
            enqueue(state,identifier,event_text(event,s['plan'],config,now))
    state['market_data_health']=dict(checked_at=now.isoformat(),feed=config['feed'],stale_symbols=stale)
    if stale:
        bucket=now.floor('30min').isoformat()
        enqueue(state,session+':stale:'+bucket,'*Price alerts paused for stale observations:* '+', '.join(stale)+f'\nFeed: {config["feed_label"]}. Fresh data is required for new entry signals.')


def recap(state,session,now):
    signals=[s for s in state.get('signals',{}).values() if s['plan']['trade_session']<=session and any(pd.Timestamp(e['at']).tz_convert('America/New_York').date().isoformat()==session for e in s['history'])]
    lines=[f'*Market-close signal recap | {session}*']
    lines += [f'{s["plan"]["ticker"]} · {s["plan"]["side"]} · {s["status"].replace("_"," ")}' for s in signals]
    if not signals:lines.append('No entry or exit conditions were observed today.')
    lines.append('Signal observations only; no broker execution or realized P&L tracked.')
    enqueue(state,session+':recap','\n'.join(lines))
    state.setdefault('jobs',{})[session+':recap']=dict(status='complete',completed_at=now.isoformat())


def tick(send=True,now=None):
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'already_running'}
        config=load(CONFIG,{})
        if not config.get('enabled'):return {'status':'disabled'}
        state=load(STATE,dict(jobs={},signals={},outbox={}))
        # A crash during an HTTP attempt must not cause a duplicate automatic post.
        for item in state.get('outbox',{}).values():
            if item['status']=='sending':item['status']='uncertain'
        current=pd.Timestamp(now or utcnow()).tz_convert('UTC')
        state.update(last_tick_at=utcnow(),pid=os.getpid(),mode='running',last_error=None)
        save_json(STATE,state)
        session=str(current.tz_convert('America/New_York').date());calendar=xcals.get_calendar('XNYS')
        try:
            if calendar.is_session(session):
                opening=calendar.session_open(session);closing=calendar.session_close(session)
                local=current.tz_convert('America/Denver')
                jobs=state.setdefault('jobs',{})
                morning_due=pd.Timestamp(session+' '+config.get('morning','06:00'),tz='America/Denver')
                update_due=pd.Timestamp(session+' '+config.get('premarket_update','06:45'),tz='America/Denver')
                # Do not backfill an already missed morning list into a live market.
                if morning_due<=local and current<opening and jobs.get(session+':morning',{}).get('status')!='complete':
                    morning(state,config,session,current);save_json(STATE,state)
                if update_due<=local and current<opening+pd.Timedelta(minutes=30) and jobs.get(session+':morning',{}).get('status')=='complete' and jobs.get(session+':premarket',{}).get('status')!='complete':
                    premarket(state,config,session,current);save_json(STATE,state)
                if opening<=current and (current<=closing+pd.Timedelta(minutes=5) or
                    current<closing+pd.Timedelta(hours=2) and jobs.get(session+':recap',{}).get('status')!='complete'):
                    monitor(state,config,session,opening,closing,current)
                if closing+pd.Timedelta(minutes=5)<=current<closing+pd.Timedelta(hours=2) and jobs.get(session+':recap',{}).get('status')!='complete':
                    recap(state,session,current)
            flush(state,config,send=send)
        except Exception as exc:
            state['last_error']=dict(type=type(exc).__name__,detail=str(exc) if isinstance(exc,ValueError) else 'Operation failed',at=utcnow())
            if calendar.is_session(session) and current<calendar.session_close(session):
                enqueue(state,session+':health:'+current.floor('30min').isoformat(),'*Trade automation needs attention:* '+state['last_error']['type']+'. Pending signals are withheld until required data is available.')
                try:flush(state,config,send=send)
                except Exception:pass
        state['last_tick_completed_at']=utcnow();save_json(STATE,state)
        return {k:state.get(k) for k in ('last_tick_at','last_tick_completed_at','last_error','pid')}


def serve():
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopped:
        try:print(json.dumps(tick(),allow_nan=False),flush=True)
        except Exception as exc:print(json.dumps(dict(error=type(exc).__name__)),flush=True)
        for _ in range(60):
            if stopped:break
            time.sleep(1)
