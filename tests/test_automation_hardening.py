import pytest
from tradingagents.pilot import automation as a


@pytest.mark.parametrize("headline, expected", [
    ("NVIDIA issued new guidance", False),
    ("TMO issued notes due 2031", False),
    ("Shake Shack pursued expansion", False),
    ("Company sued by SEC", True),
    ("Company files for bankruptcy", True),
    ("FDA rejects application", True),
    ("Company announces trading halt", True),
])
def test_material_news_word_boundaries(headline, expected):
    assert bool(a.MATERIAL_NEWS.search(headline)) is expected


from copy import deepcopy
import json
import pandas as pd
from tradingagents.pilot.earnings import earnings_rejection


@pytest.fixture
def morning_case(monkeypatch, tmp_path):
    for name, value in [('ROOT', tmp_path), ('BASE', tmp_path),
                        ('STATE', tmp_path/'state.json'), ('CONFIG', tmp_path/'config.json')]:
        monkeypatch.setattr(a, name, value)
    candidate = dict(ticker='SAMPLE', name='Sample', side='LONG', entry=100., stop=97.,
                     targets=[106.,109.], stop_loss_pct=3., target_profit_pct=[6.,9.],
                     matching_daily_techniques=[], trade_session='2026-11-10')
    rows = [dict(ticker='SAMPLE', research_queue='continuation', eligibility='eligible')]
    coverage = dict(universe_count=2000, successful_symbols=2000, benchmark_available=True, benchmark_fresh=True)
    monkeypatch.setattr(a, '_load_rows', lambda *args: (rows, {}, coverage.copy()))
    monkeypatch.setattr(a, '_news', lambda *args: dict(items=[], sources={'alpaca_news': {'status':'connected'}}))
    monkeypatch.setattr(a, '_enrich', lambda *args: {'status':'collected'})
    monkeypatch.setattr(a, 'fetch_news', lambda *args: ([], {'truncated':False}))
    monkeypatch.setattr(a, 'build_plans', lambda *args, **kwargs: dict(plans=[deepcopy(candidate)], rejected={}))
    monkeypatch.setattr(a, 'queue_key', lambda row: row['ticker'])
    monkeypatch.setattr(a, 'earnings_dates', lambda *args: ['2027-01-15'])
    return {}, dict(feed_label='IEX single-exchange'), pd.Timestamp('2026-11-10T13:00:00Z')


@pytest.mark.parametrize('dates,reason', [
    (['2026-11-10'], 'earnings_within_2_sessions'),
    (['2026-11-12'], 'earnings_within_2_sessions'),
    (['2026-11-13'], None),
    ([], 'earnings_date_unknown'),
])
def test_morning_earnings_gate(morning_case, monkeypatch, tmp_path, dates, reason):
    state, config, now = morning_case
    monkeypatch.setattr(a, 'earnings_dates', lambda *args: dates)
    a.morning(state, config, '2026-11-10', now)
    result = json.loads((tmp_path/'research/2026-11-10/plans.json').read_text())
    assert result['rejected'].get('SAMPLE') == reason
    assert bool(result['plans']) == (reason is None)


def test_earnings_lookup_failure_is_unknown(morning_case, monkeypatch, tmp_path):
    def fail(*args):
        raise TimeoutError()
    monkeypatch.setattr(a, 'earnings_dates', fail)
    state, config, now = morning_case
    a.morning(state, config, '2026-11-10', now)
    result = json.loads((tmp_path/'research/2026-11-10/plans.json').read_text())
    assert result['rejected']['SAMPLE'] == 'earnings_date_unknown'


def test_earnings_window_respects_holiday_and_date_range():
    cal = a.xcals.get_calendar('XNYS')
    assert earnings_rejection(['2026-11-30'], '2026-11-25', cal) == 'earnings_within_2_sessions'
    assert earnings_rejection(['2026-12-01'], '2026-11-25', cal) is None
    assert earnings_rejection(['2026-11-24', '2026-11-27'], '2026-11-25', cal) == 'earnings_within_2_sessions'
    assert earnings_rejection(['2026-10-01'], '2026-11-25', cal) == 'earnings_date_unknown'


@pytest.mark.parametrize('failures,spy_failed,raises', [(5,False,False),(50,False,True),(0,True,True)])
def test_symbol_failure_budget(morning_case, monkeypatch, tmp_path, failures, spy_failed, raises):
    state, config, now = morning_case
    errors = {f'BAD{i}':'unavailable' for i in range(failures)}
    if spy_failed:
        errors['SPY'] = 'unavailable'
    rows = [dict(ticker='SAMPLE', research_queue='continuation', eligibility='eligible')]
    coverage = dict(universe_count=2000, successful_symbols=2000-failures, benchmark_fresh=not spy_failed)
    monkeypatch.setattr(a, '_load_rows', lambda *args: (rows, errors, coverage))
    if raises:
        with pytest.raises(ValueError):
            a.morning(state, config, '2026-11-10', now)
        assert not state.get('signals')
    else:
        a.morning(state, config, '2026-11-10', now)
        assert state['signals']['2026-11-10:SAMPLE']['status'] == 'pending'
        assert 'Coverage: 1,995/2,000 symbols (5 unavailable)' in state['outbox']['2026-11-10:morning']['text']
    saved = json.loads((tmp_path/'research/2026-11-10/coverage.json').read_text())
    assert saved['errors'] == errors


def test_missing_fresh_benchmark_rejected(morning_case, monkeypatch):
    state, config, now = morning_case
    monkeypatch.setattr(a, '_load_rows', lambda *args: ([], {}, dict(universe_count=2000, benchmark_available=True, benchmark_fresh=False)))
    with pytest.raises(ValueError, match='SPY'):
        a.morning(state, config, '2026-11-10', now)



def test_premarket_failure_still_monitors_active_stop(morning_case, monkeypatch, tmp_path):
    state, config, now = morning_case
    config.update(enabled=True, feed='iex')
    a.save_json(a.CONFIG, config)
    signal = a.new_state(dict(ticker='SAMPLE', side='LONG', entry=100., stop=97., targets=[106.,109.], trade_session='2026-11-10'))
    signal['status'] = 'active'
    state.update(signals={'2026-11-10:SAMPLE':signal}, jobs={'2026-11-10:morning':{'status':'complete'}})
    a.save_json(a.STATE, state)
    def fail(*args):
        raise ValueError('News unavailable')
    monkeypatch.setattr(a, 'premarket', fail)
    monkeypatch.setattr(a, 'fetch_intraday', lambda *args, **kwargs: {'records':{'SAMPLE':[
        dict(event_at='2026-11-10T14:34:00Z', available_at='2026-11-10T14:35:00Z',
             open=100., low=96., high=100., close=98.)]}})
    monkeypatch.setattr(a.requests, 'post', lambda *args, **kwargs: pytest.fail('No HTTP posts during tests'))
    a.tick(send=False, now='2026-11-10T14:35:00Z')
    saved = a.load(a.STATE, {})
    assert saved['signals']['2026-11-10:SAMPLE']['status'] == 'stopped'
    assert '2026-11-10:SAMPLE:STOP_LEVEL_OBSERVED' in saved['outbox']
    assert saved['job_errors']['premarket']['type'] == 'ValueError'
    assert saved['last_error']['job'] == 'premarket'
    assert any(':health:' in key for key in saved['outbox'])


def test_monitor_failure_still_runs_recap_and_delivery(morning_case, monkeypatch):
    state, config, now = morning_case
    config['enabled'] = True
    a.save_json(a.CONFIG, config)
    calls = []
    def fail(*args):
        raise ValueError('Market feed unavailable')
    monkeypatch.setattr(a, 'monitor', fail)
    monkeypatch.setattr(a, 'recap', lambda *args: calls.append('recap'))
    monkeypatch.setattr(a, 'flush', lambda *args, **kwargs: calls.append('flush'))
    a.tick(send=False, now='2026-11-10T21:05:00Z')
    assert calls == ['recap', 'flush']
    assert a.load(a.STATE, {})['job_errors']['monitor']['type'] == 'ValueError'


from types import SimpleNamespace
import hashlib


@pytest.fixture
def delivery_case(monkeypatch, tmp_path):
    monkeypatch.setattr(a, 'STATE', tmp_path/'state.json')
    monkeypatch.setattr(a, 'slack_webhook', lambda: 'fake hook')
    clock = [pd.Timestamp('2026-11-10T13:00:00Z')]
    monkeypatch.setattr(a, 'utcnow', lambda: clock[0].isoformat())
    config = {'destination_sha256':hashlib.sha256(b'fake hook').hexdigest()}
    state = {}
    a.enqueue(state, 'test', 'message')
    return state, config, clock


@pytest.mark.parametrize('status', [429,503])
def test_retry_waits_then_succeeds(delivery_case, monkeypatch, status):
    state, config, clock = delivery_case
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(status_code=status if len(calls)==1 else 200, text='ok', headers={'Retry-After':'120'})
    monkeypatch.setattr(a.requests, 'post', post)
    a.flush(state, config)
    assert state['outbox']['test']['status'] == 'pending'
    clock[0] += pd.Timedelta(seconds=119)
    a.flush(state, config)
    assert len(calls) == 1
    clock[0] += pd.Timedelta(seconds=1)
    a.flush(state, config)
    assert state['outbox']['test']['status'] == 'sent'
    assert state['outbox']['test']['attempts'] == 2


def test_retry_gives_up_after_five_attempts(delivery_case, monkeypatch):
    state, config, clock = delivery_case
    calls = []
    monkeypatch.setattr(a.requests, 'post', lambda *args, **kwargs: calls.append(1) or SimpleNamespace(status_code=500,text='',headers={}))
    for _ in range(7):
        a.flush(state, config)
        clock[0] += pd.Timedelta(seconds=60)
    assert len(calls) == 5
    assert state['outbox']['test']['status'] == 'rejected'


def test_network_exception_stays_uncertain(delivery_case, monkeypatch):
    state, config, clock = delivery_case
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise a.requests.Timeout()
    monkeypatch.setattr(a.requests, 'post', fail)
    a.flush(state, config)
    clock[0] += pd.Timedelta(hours=1)
    a.flush(state, config)
    assert len(calls) == 1
    assert state['outbox']['test']['status'] == 'uncertain'


def test_retry_http_date_header_and_permanent_rejection(delivery_case, monkeypatch):
    state, config, clock = delivery_case
    response = SimpleNamespace(headers={'Retry-After':'Tue, 10 Nov 2026 13:02:00 GMT'})
    assert a.retry_time(response, clock[0]) == clock[0]+pd.Timedelta(minutes=2)
    monkeypatch.setattr(a.requests, 'post', lambda *args, **kwargs: SimpleNamespace(status_code=403,text='',headers={}))
    a.flush(state, config)
    assert state['outbox']['test']['status'] == 'rejected'


@pytest.mark.parametrize('missing,can_arm', [({2,7},True), ({2,4,7,9},False), ({2,14},False)])
def test_sparse_confirmation_requires_threshold_and_last_minute(missing, can_arm):
    opening = pd.Timestamp('2026-11-10T14:30:00Z')
    closing = opening+pd.Timedelta(hours=6.5)
    state = a.new_state(dict(ticker='SAMPLE', side='LONG', entry=100., stop=97., targets=[106.,109.]))
    rows = [dict(event_at=(opening+pd.Timedelta(minutes=i)).isoformat(),
                 available_at=(opening+pd.Timedelta(minutes=i+1)).isoformat(),
                 open=99.5,low=99.,high=100.5,close=100.2) for i in range(15) if i not in missing]
    a.advance(state, rows, opening, closing, opening+pd.Timedelta(minutes=15))
    assert (state['status']=='armed') is can_arm
    if can_arm:
        assert state['history'][-1]['confirmation_coverage'] == '13/15 IEX minutes'
        rows.append(dict(event_at=(opening+pd.Timedelta(minutes=15)).isoformat(),
                         available_at=(opening+pd.Timedelta(minutes=16)).isoformat(),
                         open=100.,low=99.95,high=100.2,close=100.1))
        events = a.advance(state, rows, opening, closing, opening+pd.Timedelta(minutes=16))
        assert events[0]['confirmation_coverage'] == '13/15 IEX minutes'
        assert '13/15 IEX minutes' in a.event_text(events[0], state['plan'], {'feed_label':'IEX'}, opening+pd.Timedelta(minutes=16))


def test_sparse_confirmation_requires_fresh_last_minute():
    opening = pd.Timestamp('2026-11-10T14:30:00Z')
    state = a.new_state(dict(ticker='SAMPLE', side='LONG', entry=100., stop=97., targets=[106.,109.]))
    rows = [dict(event_at=(opening+pd.Timedelta(minutes=i)).isoformat(),
                 available_at=(opening+pd.Timedelta(minutes=i+1)).isoformat(),
                 open=99.5,low=99.,high=100.5,close=100.2) for i in range(15)]
    a.advance(state, rows, opening, opening+pd.Timedelta(hours=6.5), opening+pd.Timedelta(minutes=18,seconds=1))
    assert state['status'] == 'pending'


@pytest.mark.parametrize('age,stale', [(599,False),(601,True)])
def test_monitor_stale_threshold_separate_from_entry_freshness(morning_case, monkeypatch, age, stale):
    state, config, now = morning_case
    config['feed'] = 'iex'
    opening = pd.Timestamp('2026-11-10T14:30:00Z')
    now = opening+pd.Timedelta(hours=1)
    signal = a.new_state(dict(ticker='SAMPLE',side='LONG',entry=100.,stop=97.,targets=[106.,109.],trade_session='2026-11-10'))
    state['signals'] = {'2026-11-10:SAMPLE':signal}
    available = now-pd.Timedelta(seconds=age)
    monkeypatch.setattr(a, 'fetch_intraday', lambda *args, **kwargs: {'records':{'SAMPLE':[
        dict(event_at=(available-pd.Timedelta(minutes=1)).isoformat(),available_at=available.isoformat(),
             open=99.,low=98.,high=99.5,close=99.)]}})
    a.monitor(state,config,'2026-11-10',opening,opening+pd.Timedelta(hours=6.5),now)
    assert bool(state['market_data_health']['stale_symbols']) is stale



def test_shadow_suppresses_new_and_previously_queued_trade_items(delivery_case, monkeypatch):
    state, config, clock = delivery_case
    state['outbox'].clear()
    config['shadow_mode'] = True
    a.enqueue(state, '2026-11-10:morning', 'plan')
    a.enqueue(state, '2026-11-10:SAMPLE:ENTRY_SIGNAL_OBSERVED', 'entry', kind='event', shadow=True)
    monkeypatch.setattr(a.requests, 'post', lambda *args, **kwargs: pytest.fail('Shadow trade items must never post'))
    a.flush(state, config)
    assert all(item['status']=='shadow' for item in state['outbox'].values())
    config['shadow_mode'] = False
    a.flush(state, config)
    assert all(item['status']=='shadow' for item in state['outbox'].values())


def test_shadow_keeps_health_and_coverage_delivery(delivery_case, monkeypatch):
    state, config, clock = delivery_case
    state['outbox'].clear()
    config['shadow_mode'] = True
    for event, text in [('2026-11-10:morning','plan'),('2026-11-10:coverage','coverage'),('2026-11-10:health:bucket','health')]:
        a.enqueue(state, event, text)
    posted = []
    monkeypatch.setattr(a.requests, 'post', lambda *args, **kwargs: posted.append(kwargs['json']['text']) or SimpleNamespace(status_code=200,text='ok'))
    a.flush(state, config)
    assert posted == ['coverage','health']
    a.recap(state, '2026-11-10', clock[0], config)
    recap = state['outbox']['2026-11-10:recap']
    assert recap['status'] == 'pending'
    assert 'SHADOW — not posted as plans' in recap['text']
    assert 'Would have sent:\nplan' in recap['text']


def test_shadow_morning_records_plan_and_separate_coverage(morning_case):
    state, config, now = morning_case
    config['shadow_mode'] = True
    a.morning(state, config, '2026-11-10', now)
    assert state['outbox']['2026-11-10:morning']['status'] == 'shadow'
    assert state['outbox']['2026-11-10:coverage']['status'] == 'pending'
    assert state['signals']['2026-11-10:SAMPLE']['status'] == 'pending'
