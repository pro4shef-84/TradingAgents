import pandas as pd
import pytest
from scripts import validate_specialists as v


def bars():
    return pd.DataFrame({'Open':[100,100,120,150,160,170], 'High':[110,110,130,160,170,180],
        'Low':[90,90,110,140,150,160],'Close':[100,110,125,155,165,175], 'Volume':[1000]*6},
        index=pd.bdate_range('2026-01-05',periods=6))


def signal(prefix,asof=None):
    return {'agents':[dict(id=5,status='matched',direction='bullish',evidence={'rule_version':'v1'}),
                      dict(id=50,status='matched',direction='bearish')]}


def test_next_open_fixed_exit_cost_and_split(monkeypatch):
    monkeypatch.setattr(v,'evaluate_stock',signal)
    out=v.evaluate_history({'SPY':bars()},formation_end='2026-01-06',start='2026-01-05',end='2026-01-12',holding_bars=2,cost_bps=20)
    record=next(r for r in out['observations'] if r['signal_at']=='2026-01-07')
    assert record['entry_at']=='2026-01-08' and record['entry_open']==150
    assert record['exit_at']=='2026-01-09' and record['exit_close']==165
    assert record['net_return']==pytest.approx(.098)
    assert record['split']=='evaluation'
    assert all(r['rule_id']==5 for r in out['observations'])
    assert all(r['exit_at']<='2026-01-06' for r in out['observations'] if r['split']=='formation')


def test_prefix_has_no_future_signal_and_missing_exit_not_invented(monkeypatch):
    seen=[]
    def capture(prefix,asof=None):
        assert str(prefix.index[-1].date())==asof
        seen.append(len(prefix))
        return signal(prefix,asof)
    monkeypatch.setattr(v,'evaluate_stock',capture)
    original=bars()
    out=v.evaluate_history({'SPY':original},'2026-01-06','2026-01-05','2026-01-08',holding_bars=1)
    altered=original.copy();altered.loc['2026-01-09':,'Close']=999
    again=v.evaluate_history({'SPY':altered},'2026-01-06','2026-01-05','2026-01-08',holding_bars=1)
    assert [r for r in out['observations'] if r['exit_at']<='2026-01-08']==[r for r in again['observations'] if r['exit_at']<='2026-01-08']
    assert max(seen)==4
    assert all(r['exit_at']<='2026-01-08' for r in out['observations'])
    assert out['limitations']['point_in_time_supplemental']=='absent'
    assert out['limitations']['surviving_universe_bias']


def test_inputs_reject_noncausal_or_invalid_dates():
    with pytest.raises(ValueError):v.evaluate_history({'SPY':bars().iloc[::-1]},'2026-01-06','2026-01-05','2026-01-12')
    with pytest.raises(ValueError):v.evaluate_history({'SPY':bars()},'2026-01-06','2026-01-05','2026-01-12',cost_bps=-1)


def test_spy_is_matched_to_entry_and_exit_and_receives_same_cost(monkeypatch):
    monkeypatch.setattr(v,'evaluate_stock',signal)
    stock=bars()
    spy=bars().copy()
    spy.loc['2026-01-08','Open']=100
    spy.loc['2026-01-08','Low']=90
    spy.loc['2026-01-09','Close']=170
    out=v.evaluate_history({'SPY':spy,'A':stock},'2026-01-06','2026-01-05','2026-01-12',holding_bars=2,cost_bps=20)
    r=next(r for r in out['observations'] if r['ticker']=='A' and r['signal_at']=='2026-01-07')
    assert r['spy_return']==pytest.approx(.698)
    assert r['excess_return']==pytest.approx(-.6)
    assert not any(r['signal_at']<='2026-01-06'<r['exit_at'] for r in out['observations'])


def test_missing_benchmark_is_not_counted_as_outperformance(monkeypatch):
    monkeypatch.setattr(v,'evaluate_stock',signal)
    out=v.evaluate_history({'A':bars()},'2026-01-06','2026-01-05','2026-01-12',holding_bars=1)
    assert all(r['excess_return'] is None for r in out['observations'])
    assert all(r['benchmark_pairs']==0 and not r['selected_in_formation'] for r in out['summaries'])


def test_future_prices_do_not_change_formation_selection(monkeypatch):
    monkeypatch.setattr(v,'evaluate_stock',signal)
    stock=pd.concat([bars()]*12,ignore_index=True)
    stock.index=pd.bdate_range('2026-01-05',periods=len(stock))
    spy=stock.copy()
    spy[['Open','High','Low','Close']]=100.
    cutoff=str(stock.index[39].date())
    args=dict(formation_end=cutoff,start='2026-01-05',end=str(stock.index[-1].date()),holding_bars=1)
    first=v.evaluate_history({'A':stock,'SPY':spy},**args)
    changed=stock.copy()
    changed.loc[changed.index>cutoff,['Open','High','Low','Close']]*=2
    second=v.evaluate_history({'A':changed,'SPY':spy},**args)
    assert [r for r in first['summaries'] if r['split']=='formation']==[r for r in second['summaries'] if r['split']=='formation']
    assert any(r['selected_in_formation'] for r in first['summaries'] if r['rule_id']==5)


def test_cohort_comparison_deduplicates_rules_and_filters_formation_selection():
    window=dict(ticker='A',signal_at='2026-07-01',entry_at='2026-07-02',
                direction='bullish',split='evaluation',net_return=.01,
                spy_return=.02,excess_return=-.01)
    result=dict(protocol={'shortlist':['A'],'control':[]},
        summaries=[dict(split='formation',rule_id=5,direction='bullish',selected_in_formation=True)],
        observations=[window|{'rule_id':5},window|{'rule_id':6},
                      window|{'rule_id':6,'signal_at':'2026-07-02'}])
    comparison=v.compare_cohorts(result)
    assert comparison[0]['windows']==2
    assert comparison[1]['windows']==1
    assert comparison[1]['mean_excess_return']==pytest.approx(-.01)
    assert comparison[2]['mean_net_return'] is None
