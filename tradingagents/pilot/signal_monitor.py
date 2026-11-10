"""Observed signal state transitions. No orders, fill claims or profit attribution."""
import pandas as pd

TERMINAL={'stopped','completed','expired','review_required','time_exit'}


def new_state(plan):
    return dict(plan=plan,status='pending',armed_at=None,entry_at=None,entry_observation=None,
                tp1=False,last_bar_at=None,history=[])


def advance(state,records,opening,closing,now,fresh_seconds=180,allow_entries=True):
    """Consume completed regular-session minutes in order, emitting new observations."""
    now=pd.Timestamp(now);opening=pd.Timestamp(opening);closing=pd.Timestamp(closing)
    if state['status'] in TERMINAL:return []
    plan=state['plan'];sign=1 if plan['side']=='LONG' else -1
    entry,stop=plan['entry'],plan['stop'];tp1,tp2=plan['targets']
    rows=sorted([r for r in records if opening<=pd.Timestamp(r['event_at'])<closing
                 and pd.Timestamp(r['available_at'])<=now],key=lambda r:r['event_at'])
    by_time={pd.Timestamp(r['event_at']):r for r in rows}
    events=[]
    def emit(kind,bar,**details):
        at=bar['available_at'] if bar else now.isoformat()
        item=dict(kind=kind,ticker=plan['ticker'],side=plan['side'],at=at,
                  reference_entry=entry,**details)
        state['history'].append(item);events.append(item)
    for t,bar in sorted(by_time.items()):
        if state['last_bar_at'] and t<=pd.Timestamp(state['last_bar_at']):continue
        state['last_bar_at']=t.isoformat()
        low,high,price=float(bar['low']),float(bar['high']),float(bar['close'])
        if state['status']=='active':
            stop_touch=low<=stop if sign==1 else high>=stop
            target_touch=high>=tp1 if sign==1 else low<=tp1
            target2_touch=high>=tp2 if sign==1 else low<=tp2
            if stop_touch and (target_touch and not state['tp1'] or target2_touch):
                state['status']='review_required';emit('AMBIGUOUS_INTERVAL',bar,stop=stop,targets=[tp1,tp2]);break
            if stop_touch:
                state['status']='stopped';emit('STOP_LEVEL_OBSERVED',bar,level=stop,bar_open=bar['open']);break
            if target_touch and not state['tp1']:
                state['tp1']=True;emit('TP1_LEVEL_OBSERVED',bar,level=tp1,planned_exit_pct=50)
            if target2_touch:
                state['status']='completed';emit('TP2_LEVEL_OBSERVED',bar,level=tp2,planned_exit_pct=50);break
            continue
        # No stale catch-up entry is presented as actionable. Past missed bars can
        # still establish an arm, but the observed retest must itself be fresh.
        fresh=(now-pd.Timestamp(bar['available_at'])).total_seconds()<=fresh_seconds
        if state['status']=='armed' and t-pd.Timestamp(state['armed_at'])>pd.Timedelta(minutes=30):
            state.update(status='pending',armed_at=None)
        if state['status']=='armed' and state['armed_at'] and t>=pd.Timestamp(state['armed_at']):
            tolerance=max(.01,entry*.0025)
            retest=low<=entry<=high and 0<=sign*(price-entry)<=tolerance
            if retest and fresh and allow_entries:
                if (low<=stop if sign==1 else high>=stop) or (high>=tp1 if sign==1 else low<=tp1):
                    state['status']='review_required';emit('AMBIGUOUS_ENTRY_INTERVAL',bar);break
                state.update(status='active',entry_at=bar['available_at'],entry_observation=price)
                emit('ENTRY_SIGNAL_OBSERVED',bar,price=price);continue
        elapsed=int((t+pd.Timedelta(minutes=1)-opening).total_seconds()/60)
        if elapsed<15 or elapsed%15:continue
        start=t-pd.Timedelta(minutes=14)
        expected=pd.date_range(start,t,freq='min')
        if not all(x in by_time for x in expected):continue
        block=[by_time[x] for x in expected]
        cross=min(b['low'] for b in block)<=entry<=max(b['high'] for b in block)
        if cross and sign*(price-entry)>0:
            state.update(status='armed',armed_at=bar['available_at'])
    # Pending plans expire on their activation session; active observations carry.
    if now>=closing and state['status'] in ('pending','armed'):
        state['status']='expired';emit('SETUP_EXPIRED',None)
    return events
