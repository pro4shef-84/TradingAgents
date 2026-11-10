from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.pilot.signal_monitor import advance,new_state
from tradingagents.pilot.trade_planner import make_plan,active_matches
from tradingagents.pilot import automation as a

OPEN=pd.Timestamp('2026-11-10T14:30:00Z')
CLOSE=pd.Timestamp('2026-11-10T21:00:00Z')


def plan(side='LONG'):
    return dict(ticker='SAMPLE',side=side,entry=100.,stop=97. if side=='LONG' else 103.,
                targets=[106.,109.] if side=='LONG' else [94.,91.],trade_session='2026-11-10')


def bar(n,**updates):
    t=OPEN+pd.Timedelta(minutes=n)
    return dict(event_at=t.isoformat(),available_at=(t+pd.Timedelta(minutes=1)).isoformat(),
                open=99.5,high=100.5,low=99.,close=100.2,**{})|updates


def confirmed():
    return [bar(i) for i in range(15)]+[bar(15,low=99.95,high=100.2,close=100.1)]


def test_entry_requires_completed_confirmation_then_later_retest():
    state=new_state(plan());rows=confirmed()
    assert advance(state,rows[:15],OPEN,CLOSE,OPEN+pd.Timedelta(minutes=15))==[]
    assert state['status']=='armed'
    events=advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=16))
    assert [e['kind'] for e in events]==['ENTRY_SIGNAL_OBSERVED']
    assert state['entry_at']==(OPEN+pd.Timedelta(minutes=16)).isoformat()
    assert advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=16))==[]


def test_insufficient_confirmation_minutes_or_future_bar_cannot_trigger():
    rows=confirmed();state=new_state(plan())
    advance(state,rows[:5]+rows[9:],OPEN,CLOSE,OPEN+pd.Timedelta(minutes=16))
    assert state['status']=='pending'
    state=new_state(plan())
    advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=14))
    assert state['status']=='pending'


def test_stale_catchup_cannot_emit_actionable_entry():
    state=new_state(plan())
    assert advance(state,confirmed(),OPEN,CLOSE,OPEN+pd.Timedelta(hours=1))==[]
    assert state['status']!='active'


def test_incomplete_news_readiness_prevents_new_entry():
    state=new_state(plan())
    assert advance(state,confirmed(),OPEN,CLOSE,OPEN+pd.Timedelta(minutes=16),allow_entries=False)==[]
    assert state['status']!='active'


def test_same_minute_stop_and_target_has_no_invented_order():
    state=new_state(plan());rows=confirmed()
    advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=16))
    events=advance(state,rows+[bar(16,low=96.,high=110.)],OPEN,CLOSE,OPEN+pd.Timedelta(minutes=17))
    assert state['status']=='review_required'
    assert [e['kind'] for e in events]==['AMBIGUOUS_INTERVAL']


def test_no_target_attribution_from_before_entry_and_no_duplicate_target():
    rows=confirmed();rows[3]['high']=110.
    state=new_state(plan())
    assert [e['kind'] for e in advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=16))]==['ENTRY_SIGNAL_OBSERVED']
    rows+=[bar(16,high=106.5,low=100.,close=105.)]
    assert [e['kind'] for e in advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=17))]==['TP1_LEVEL_OBSERVED']
    assert advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=17))==[]


def test_short_direction_and_expiry():
    state=new_state(plan('SHORT'))
    rows=[bar(i,close=99.8,high=101.,low=99.5) for i in range(15)]
    rows+=[bar(15,close=99.9,high=100.05,low=99.8)]
    assert [e['kind'] for e in advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=16))]==['ENTRY_SIGNAL_OBSERVED']
    rows+=[bar(16,open=99.,high=99.5,low=93.5,close=94.)]
    assert [e['kind'] for e in advance(state,rows,OPEN,CLOSE,OPEN+pd.Timedelta(minutes=17))]==['TP1_LEVEL_OBSERVED']
    untouched=new_state(plan())
    assert advance(untouched,[],OPEN,CLOSE,CLOSE)[0]['kind']=='SETUP_EXPIRED'


def test_slack_delivery_success_and_uncertainty_are_never_automatically_retried(monkeypatch,tmp_path):
    import hashlib
    monkeypatch.setattr(a,'STATE',tmp_path/'state.json')
    monkeypatch.setattr(a,'slack_webhook',lambda:'fake webhook')
    config={'destination_sha256':hashlib.sha256(b'fake webhook').hexdigest()}
    calls=[]
    monkeypatch.setattr(a.requests,'post',lambda *args,**kw:calls.append(1) or SimpleNamespace(status_code=200,text='ok'))
    state={};a.enqueue(state,'event','hello');a.enqueue(state,'event','duplicate')
    a.flush(state,config);a.flush(state,config)
    assert len(calls)==1 and state['outbox']['event']['status']=='sent'
    def timeout(*args,**kwargs):
        calls.append(1);raise a.requests.Timeout()
    monkeypatch.setattr(a.requests,'post',timeout)
    a.enqueue(state,'next','next');a.flush(state,config);a.flush(state,config)
    assert len(calls)==2 and state['outbox']['next']['status']=='uncertain'


def fixture():
    index=pd.bdate_range(end='2026-11-09',periods=130)
    bars=pd.DataFrame(dict(Open=100.,High=101.,Low=99.,Close=100.,Volume=1e6),index=index)
    bars.iloc[-1]=[100.,103.,100.,102.,1e6]
    agents=[dict(id=i,name=f'Rule {i}',family=family,status='matched',direction='bullish',
                 events=[dict(direction='bullish',level=102.)]) for i,family in enumerate(('trend','volume','momentum'))]
    return dict(ticker='SAMPLE',name='Sample Systems',close=102.,asof='2026-11-09',
                context={'atr':2.,'rvol':1.5},agents=agents),bars


def test_planner_prices_and_percentages_follow_direction_and_chart():
    row,bars=fixture();p,reason=make_plan(row,bars,'2026-11-10')
    assert reason is None and p['stop']<p['entry']<p['targets'][0]<p['targets'][1]
    assert p['target_R'][0]==pytest.approx(2.)
    assert p['stop_loss_pct']==pytest.approx((p['entry']-p['stop'])/p['entry']*100)


def test_nearby_resistance_or_invalidated_evidence_prevents_plan():
    row,bars=fixture();bars.iloc[80,bars.columns.get_loc('High')]=105.
    assert make_plan(row,bars,'2026-11-10')[1]=='nearby_obstacle_limits_reward'
    row,bars=fixture();row['agents'][0]['events'][0]['invalidation']=103.
    assert len(active_matches(row))==2
    assert make_plan(row,bars,'2026-11-10')[0] is None


def test_destination_change_blocks_post(monkeypatch,tmp_path):
    monkeypatch.setattr(a,'slack_webhook',lambda:'changed webhook')
    state={};a.enqueue(state,'event','message')
    with pytest.raises(ValueError,match='destination changed'):a.flush(state,{'destination_sha256':'original'})


def test_schedule_uses_mountain_time_and_jobs_run_once(monkeypatch,tmp_path):
    monkeypatch.setattr(a,'BASE',tmp_path)
    monkeypatch.setattr(a,'STATE',tmp_path/'state.json')
    monkeypatch.setattr(a,'CONFIG',tmp_path/'config.json')
    (tmp_path/'config.json').write_text(json.dumps({'enabled':True}))
    calls=[]
    def job(name):
        def run(state,config,session,now):
            calls.append(name);state.setdefault('jobs',{})[session+':'+name]={'status':'complete'}
        return run
    monkeypatch.setattr(a,'morning',job('morning'))
    monkeypatch.setattr(a,'premarket',job('premarket'))
    monkeypatch.setattr(a,'flush',lambda *args,**kwargs:None)
    a.tick(send=False,now='2026-11-10T12:59:00Z')
    assert calls==[]
    a.tick(send=False,now='2026-11-10T13:00:00Z')
    a.tick(send=False,now='2026-11-10T13:01:00Z')
    assert calls==['morning']
    a.tick(send=False,now='2026-11-10T13:45:00Z')
    assert calls==['morning','premarket']
    a.tick(send=False,now='2026-11-14T13:00:00Z')
    assert calls==['morning','premarket']
