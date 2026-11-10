"""Observed signal state transitions. No orders, fill claims or profit attribution."""
import pandas as pd

TERMINAL={'stopped','completed','expired','review_required','time_exit'}


def new_state(plan):
    return dict(plan=plan,status='pending',armed_at=None,entry_at=None,entry_observation=None,
                tp1=False,last_bar_at=None,history=[])


def advance(state,records,opening,closing,now,fresh_seconds=180,allow_entries=True,min_confirmation_bars=12,feed_label='IEX'):
    """Consume completed regular-session minutes in order, emitting new observations."""
    if not isinstance(min_confirmation_bars,int) or not 1<=min_confirmation_bars<=15:
        raise ValueError('min_confirmation_bars must be an integer from 1 to 15')
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
        # No stale catch-up entry is presented as actionable. Both the confirmation closing minute
        # and the later observed retest must themselves be fresh.
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
                emit('ENTRY_SIGNAL_OBSERVED',bar,price=price,confirmation_coverage=state.get('confirmation_coverage'));continue
        elapsed=int((t+pd.Timedelta(minutes=1)-opening).total_seconds()/60)
        if elapsed<15 or elapsed%15:continue
        start=t-pd.Timedelta(minutes=14)
        expected=pd.date_range(start,t,freq='min')
        block=[by_time[x] for x in expected if x in by_time]
        if len(block)<min_confirmation_bars or not fresh:continue
        cross=min(b['low'] for b in block)<=entry<=max(b['high'] for b in block)
        if cross and sign*(price-entry)>0:
            coverage=f'{len(block)}/15 {feed_label} minutes'
            state.update(status='armed',armed_at=bar['available_at'],confirmation_coverage=coverage)
            state['history'].append(dict(kind='CONFIRMATION_OBSERVED',ticker=plan['ticker'],side=plan['side'],
                                         at=bar['available_at'],confirmation_coverage=coverage))
    # Pending plans expire on their activation session; active observations carry.
    if now>=closing and state['status'] in ('pending','armed'):
        state['status']='expired';emit('SETUP_EXPIRED',None)
    return events
