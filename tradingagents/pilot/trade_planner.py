"""Versioned, deterministic chart plans. Targets are references, never forecasts."""
import math
import re
from pathlib import Path

import pandas as pd

from tradingagents.pilot.screen import queue_key

VERSION='chart_plans_v1'
RULES=dict(min_families=3,min_reward_risk=2.,max_risk_pct=5.,
           min_stop_atr=.5,max_stop_atr=1.5,max_daily_move_pct=10.,
           max_rebound_atr=1.75,max_per_side=2)


def active_matches(row):
    result=[];price=row['close']
    for agent in row.get('agents',[]):
        direction=agent.get('direction')
        if agent['status']!='matched' or direction not in ('bullish','bearish'):continue
        events=[e for e in agent.get('events',[]) if e.get('direction')==direction]
        if not events:continue
        valid=[]
        for event in events:
            invalidation=event.get('invalidation')
            if invalidation is not None and ((direction=='bullish' and price<=invalidation) or
                                             (direction=='bearish' and price>=invalidation)):continue
            valid.append(event)
        if valid:result.append({**agent,'events':valid})
    return result


def make_plan(row,bars,session):
    agents=active_matches(row)
    directions={a['direction'] for a in agents}
    if len(directions)!=1:return None,'conflicting_or_no_active_evidence'
    direction=next(iter(directions));sign=1 if direction=='bullish' else -1
    if len({a['family'] for a in agents})<RULES['min_families']:return None,'too_few_evidence_families'
    # Automatic posts avoid binary clinical-event names; they remain in research.
    if re.search(r'pharma|therapeutic|oncology|biotech|immunitybio',row.get('name',''),re.I):
        return None,'clinical_event_name_requires_manual_review'
    bars=bars.loc[:row['asof']]
    if len(bars)<65:return None,'insufficient_history'
    last=bars.iloc[-1];atr=row['context'].get('atr')
    if not atr or not math.isfinite(atr) or atr<=0:return None,'missing_atr'
    daily_change=(last.Close/bars.Close.iloc[-2]-1)*100
    if abs(daily_change)>RULES['max_daily_move_pct']:return None,'extended_daily_move'
    buffer=max(.01,.05*atr)
    prior_high=float(bars.High.iloc[-21:-1].max());prior_low=float(bars.Low.iloc[-21:-1].min())
    if sign==1:
        entry=round(float(last.High)+.01,2)
        support=prior_high if last.Low<prior_high<entry else float(last.Low)
        stop=round(min(support-buffer,entry-RULES['min_stop_atr']*atr),2)
        trigger='15-minute close above entry, followed by a confirming retest near entry'
    else:
        if prior_low<=last.Close:return None,'no_fresh_range_breakdown'
        entry=round(prior_low-.02*atr,2)
        if entry-last.Close>RULES['max_rebound_atr']*atr:return None,'rebound_entry_too_far'
        levels=[e['invalidation'] for a in agents for e in a['events']
                if isinstance(e.get('invalidation'),(float,int)) and e['invalidation']>entry]
        levels += [float(x) for x in (last.High,bars.Low.iloc[-2]) if x>entry]
        resistance=min(levels) if levels else entry+atr
        stop=round(max(resistance+buffer,entry+RULES['min_stop_atr']*atr),2)
        trigger='rebound to entry zone fails, then 15-minute close below entry and confirming retest'
    risk=abs(entry-stop)
    if risk<=0 or risk/entry*100>RULES['max_risk_pct'] or risk>RULES['max_stop_atr']*atr:
        return None,'stop_too_wide'
    # Major references from the most recent 90 sessions, confirmed on both sides.
    window=bars.tail(90);values=window.High.to_numpy() if sign==1 else window.Low.to_numpy()
    references=[]
    for i in range(3,len(values)-3):
        extremum=max(values[i-3:i+4]) if sign==1 else min(values[i-3:i+4])
        if values[i]==extremum and sign*(values[i]-entry)>0:
            if sign==1 or values[i]<=last.Close:references.append(float(values[i]))
    if sign==-1 and last.Low<entry:references.append(float(last.Low))
    references=sorted(set(references),reverse=sign==-1)
    targets=[];basis=[]
    for reference in references:
        target=round(reference-sign*min(buffer,.05*risk),2)
        if targets and abs(target-targets[-1])<.5*risk:continue
        if not targets and sign*(target-entry)/risk<RULES['min_reward_risk']:
            return None,'nearby_obstacle_limits_reward'
        targets.append(target);basis.append('confirmed 90-session chart reference')
        if len(targets)==2:break
    if not targets:
        targets=[round(entry+sign*2*risk,2),round(entry+sign*3*risk,2)]
        basis=['2R projection','3R projection']
    elif len(targets)==1:
        targets.append(round(entry+sign*max(3*risk,abs(targets[0]-entry)+risk),2))
        basis.append('risk-multiple projection')
    if any(t<=0 for t in targets):return None,'invalid_target'
    side='LONG' if sign==1 else 'SHORT'
    plan=dict(ticker=row['ticker'],name=row.get('name',''),side=side,entry=entry,stop=stop,
              targets=targets,target_basis=basis,trigger=trigger,source_daily_session=row['asof'],
              trade_session=session,status='pending_confirmation',policy=VERSION,
              stop_loss_pct=risk/entry*100,target_profit_pct=[sign*(t/entry-1)*100 for t in targets],
              target_R=[sign*(t-entry)/risk for t in targets],
              matching_daily_techniques=[dict(id=a['id'],name=a['name'],family=a['family']) for a in agents],
              target_exit_fractions=[.5,.5],atr=float(atr),volume_multiple=row['context'].get('rvol'))
    return plan,None


def build_plans(root,rows,session,exclude=()):
    plans=[];rejected={};counts={'LONG':0,'SHORT':0}
    for row in sorted(rows,key=queue_key):
        ticker=row['ticker']
        if ticker in exclude or row['eligibility'] not in ('eligible','provisional'):continue
        if row['liquidity_tier']!='preferred':continue
        if row['research_queue'] not in ('continuation','recovery','bearish'):continue
        side='SHORT' if row['research_queue']=='bearish' else 'LONG'
        if counts[side]>=RULES['max_per_side']:continue
        frame=pd.read_csv(Path(root)/row['price_file'],index_col=0,parse_dates=True)
        plan,reason=make_plan(row,frame,session)
        if plan is None:rejected[ticker]=reason;continue
        plans.append(plan);counts[plan['side']]+=1
    return dict(policy=VERSION,rules=RULES,plans=plans,rejected=rejected,
                selection='Up to two plans per direction; no padding; family-capped queue order')
