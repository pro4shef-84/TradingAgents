"""Causal fixtures for research definitions, never performance evidence."""
import copy
import json
import unittest
import numpy as np
import pandas as pd
from tradingagents.technique_panel import evaluate_stock


def bars(values):
    v = np.asarray(values, float)
    d=pd.DataFrame(dict(Open=v, High=v+.2, Low=v-.2, Close=v, Volume=1e6),
                   index=pd.bdate_range(end='2026-11-06', periods=len(v)))
    d.attrs['adjustment']='raw'
    return d


def envelope(records, session='2026-11-09', **extra):
    return dict(provider='fixture', feed='fixture', symbol='ABC', session=session,
                retrieved_at='2026-11-09T22:00:00Z', adjustment='raw',
                coverage={'complete': True}, records=records, **extra)


def intraday(values, date='2026-11-09'):
    result=[]
    for i, value in enumerate(values):
        t=pd.Timestamp(date+'T09:30:00', tz='America/New_York')+pd.Timedelta(minutes=5*i)
        result.append(dict(timestamp=t.isoformat(), available_at=(t+pd.Timedelta(minutes=5)).isoformat(),
                           interval_minutes=5,open=value, high=value+.1,low=value-.1,close=value,volume=100))
    return result


def evaluate(identifier, supplemental=None, daily=None):
    daily=bars([100]*240) if daily is None else daily
    return evaluate_stock(daily, supplemental=supplemental)['agents'][identifier-1]


class SpecialistRulesTests(unittest.TestCase):
    def test_daily_proxies_have_no_unsupported_placeholder(self):
        for identifier in (5,6,12,14,25,39):
            row=evaluate(identifier)
            self.assertEqual(row['status'],'no_match')
            self.assertEqual(row['rule_version'],'specialist_rules_v1')
        self.assertIn('Price-zone',evaluate(25)['name'])

    def test_rounding_bottom_positive_watch_negative_and_future_invariance(self):
        cup=np.r_[np.linspace(100,90,20),np.linspace(90,100,20)]
        d=bars(np.r_[[100]*100,cup,101])
        self.assertEqual(evaluate(5,daily=d)['status'],'matched')
        d.iloc[-1]=d.iloc[-2]
        self.assertEqual(evaluate(5,daily=d)['status'],'watch')
        self.assertIsNone(evaluate(5,daily=d)['evidence'][0]['confirmation_time'])
        d=bars(np.r_[[100]*100,cup,101,999])
        self.assertEqual(evaluate_stock(d,asof=d.index[-2]),evaluate_stock(d.iloc[:-1]))

    def test_gap_and_wyckoff_both_directions(self):
        d=bars(np.r_[[100]*210,np.linspace(100,110,29),113])
        d.iloc[-1,d.columns.get_loc('Volume')]=2e6
        self.assertEqual(evaluate(14,daily=d)['direction'],'bullish')
        d=bars([100]*240)
        d.iloc[-1]=[100,100.2,99,100,2e6]
        self.assertEqual(evaluate(39,daily=d)['direction'],'bullish')
        d.iloc[-1]=[100,101,99.8,100,2e6]
        self.assertEqual(evaluate(39,daily=d)['direction'],'bearish')

    def test_intraday_opening_range_and_gap_require_complete_intervals(self):
        s={'symbol':'ABC','cutoff':'2026-11-09T16:00:00Z',
           'intraday':envelope(intraday([102]*6+[103]))}
        self.assertEqual(evaluate(15,s)['direction'],'bullish')
        self.assertEqual(evaluate(46,s)['direction'],'bullish')
        self.assertEqual(evaluate(15,s)['events'][0]['timeframe'],'intraday')
        s['intraday']['records'].pop(2)
        self.assertEqual(evaluate(15,s)['status'],'missing_data')

    def test_future_and_unavailable_intraday_cannot_change_signal(self):
        s={'cutoff':'2026-11-09T15:05:00Z','intraday':envelope(intraday([100]*6+[102]))}
        expected=evaluate(15,s)
        s['intraday']['records']+=intraday([900]*30)[7:]
        self.assertEqual(evaluate(15,s),expected)
        s['intraday']['records'][6]['available_at']='2026-11-09T18:00:00Z'
        self.assertNotEqual(evaluate(15,s)['status'],'matched')

    def test_naive_timestamp_and_bad_ohlc_abstain(self):
        for field,value in [('timestamp','2026-11-09T09:30:00'),('high',1)]:
            s={'cutoff':'2026-11-09T17:00:00Z','intraday':envelope(intraday([100]*7))}
            s['intraday']['records'][0][field]=value
            self.assertEqual(evaluate(15,s)['status'],'invalid_data')

    def test_volume_profile_uses_trade_size_and_tape_uses_prior_quotes(self):
        t=pd.Timestamp('2026-11-09T15:00:00Z')
        trades=[dict(timestamp=(t+pd.Timedelta(seconds=i)).isoformat(),available_at=(t+pd.Timedelta(seconds=i)).isoformat(),price=100 if i<9 else 101,size=100 if i<9 else 1) for i in range(10)]
        quotes=[dict(timestamp=(t+pd.Timedelta(seconds=i)).isoformat(),available_at=(t+pd.Timedelta(seconds=i)).isoformat(),bid=99.8,ask=100) for i in range(10)]
        s={'cutoff':'2026-11-09T15:01:00Z','trades':envelope(trades),'quotes':envelope(quotes)}
        self.assertEqual(evaluate(38,s)['direction'],'bullish')
        self.assertEqual(evaluate(42,s)['direction'],'bullish')
        for q in quotes:
            q['timestamp']=(t+pd.Timedelta(seconds=30)).isoformat()
            q['available_at']=q['timestamp']
        self.assertEqual(evaluate(42,s)['status'],'missing_data')
        s['trades']['coverage']['complete']=False
        self.assertEqual(evaluate(38,s)['status'],'missing_data')

    def test_book_sequence_gap_requires_resnapshot(self):
        def book(seq,kind):
            return dict(timestamp='2026-11-09T15:00:00Z',available_at='2026-11-09T15:00:00Z',sequence=seq,type=kind,bids=[[100,100]],asks=[[101,10]])
        s={'cutoff':'2026-11-09T15:01:00Z','book':envelope([book(1,'snapshot'),book(3,'update')])}
        self.assertEqual(evaluate(43,s)['status'],'missing_data')
        s['book']['records'].append(book(20,'snapshot'))
        self.assertEqual(evaluate(43,s)['direction'],'bullish')

    def test_catalyst_needs_verification_and_postavailability_bars(self):
        s={'cutoff':'2026-11-09T17:00:00Z','intraday':envelope(intraday([100,100,100,102,102,102,102])),
           'events':envelope([dict(timestamp='2026-11-09T14:30:00Z',event_time='2026-11-09T14:30:00Z',available_at='2026-11-09T14:30:00Z',verified=True,source_url='https://example.com/release',direction='bullish')])}
        self.assertEqual(evaluate(47,s)['direction'],'bullish')
        s['events']['records'][0]['verified']=False
        self.assertEqual(evaluate(47,s)['status'],'missing_evidence')
        s['events']['records'][0].update(verified=True,available_at='2026-11-09T16:59:00Z')
        self.assertNotEqual(evaluate(47,s)['status'],'matched')


    def test_diamond_cup_handle_and_price_zone_positive_negative(self):
        d=bars([100]*240)
        widths=np.r_[[.5]*8,[2]*8,[1.5]*8,[.75]*8]
        d.iloc[-33:-1,d.columns.get_loc('High')]=100+widths
        d.iloc[-33:-1,d.columns.get_loc('Low')]=100-widths
        d.iloc[-1]=[102,102.2,101.8,102,1e6]
        self.assertEqual(evaluate(6,daily=d)['status'],'matched')
        d.iloc[-1]=[100,100.2,99.8,100,1e6]
        self.assertEqual(evaluate(6,daily=d)['status'],'watch')
        cup=np.r_[np.linspace(100,90,20),np.linspace(90,100,20)]
        d=bars(np.r_[[100]*100,cup,[99,98,97,97,98,98,99,99],101])
        self.assertEqual(evaluate(12,daily=d)['status'],'matched')
        d=bars([100]*240)
        d.iloc[-10]=[98,98.2,97.8,98,1e6]
        d.iloc[-1]=[98,99.2,97.8,99,1e6]
        self.assertEqual(evaluate(25,daily=d)['direction'],'bullish')
        self.assertEqual(evaluate(25)['status'],'no_match')

    def test_vwap_cross_is_approximate_without_provider_vwap(self):
        s={'cutoff':'2026-11-09T17:00:00Z','intraday':envelope(intraday([100]*6+[102]))}
        row=evaluate(30,s)
        self.assertEqual(row['direction'],'bullish')
        self.assertTrue(row['evidence'][0]['approximate'])
        for r in s['intraday']['records']:r['vwap']=r['close']
        self.assertFalse(evaluate(30,s)['evidence'][0]['approximate'])
        s['intraday']['records'][-1]['close']=100
        s['intraday']['records'][-1]['open']=100
        s['intraday']['records'][-1]['low']=99.9
        s['intraday']['records'][-1]['high']=100.1
        s['intraday']['records'][-1]['vwap']=100
        self.assertEqual(evaluate(30,s)['status'],'no_match')

    def test_market_profile_is_time_at_price_and_requires_complete_sessions(self):
        records=intraday([100]*78,'2026-11-06')+intraday([102]*78)
        s={'cutoff':'2026-11-09T21:00:00Z','intraday':envelope(records)}
        self.assertEqual(evaluate(44,s)['direction'],'bullish')
        # Volume cannot change a time-at-price profile.
        original=evaluate(44,s)
        for r in records:r['volume']*=100
        self.assertEqual(evaluate(44,s),original)
        records.pop()
        self.assertEqual(evaluate(44,s)['status'],'missing_data')

    def test_pairs_model_uses_only_formation_and_never_votes_outright(self):
        t=pd.date_range('2026-06-01',periods=66,tz='UTC')
        x=np.linspace(4,4.2,66)
        y=.5+1.2*x+np.tile([-.002,.002],33)
        y[64]+=.025
        records=[dict(timestamp=day.isoformat(),available_at=day.isoformat(),stock=float(np.exp(a)),partner=float(np.exp(b))) for day,a,b in zip(t,y,x)]
        ds=envelope(records[:65],pair=['ABC','XYZ'],cost_bps=10,borrow_assumption='unknown')
        ds['adjustment']='adjusted'
        s={'cutoff':t[64].isoformat(),'pairs':ds}
        row=evaluate(50,s)
        self.assertEqual(row['status'],'context')
        self.assertEqual(row['direction'],'neutral')
        self.assertEqual(row['evidence'][0]['pair_direction'],'short_stock_long_partner')
        self.assertFalse(row['evidence'][0]['executable_returns'])
        original=copy.deepcopy(row)
        ds['records'].append(records[-1])
        self.assertEqual(evaluate(50,s),original)
        ds['records'][64]['stock']=float(np.exp(.5+1.2*x[64]))
        self.assertEqual(evaluate(50,s)['status'],'no_match')
        ds['borrow_assumption']=None
        self.assertEqual(evaluate(50,s)['status'],'invalid_data')

    def test_each_rule_insufficient_history_missing_and_malformed(self):
        for identifier in (5,6,12,14,25,39):
            self.assertEqual(evaluate(identifier,daily=bars([100]*20))['status'],'insufficient_history')
        for identifier in (15,30,38,42,43,44,46,47,50):
            self.assertIn(evaluate(identifier)['status'],('missing_data','missing_evidence'))
            self.assertEqual(evaluate(identifier,{'cutoff':'not-a-time'})['status'],'invalid_data')


    def test_imported_verified_event_and_book_do_not_claim_complete_feed(self):
        s={'cutoff':'2026-11-09T17:00:00Z','intraday':envelope(intraday([100,100,102,102,102,102,102])),
           'events':envelope([dict(timestamp='2026-11-09T14:30:00Z',event_time='2026-11-09T14:30:00Z',available_at='2026-11-09T14:30:00Z',verified=True,source_url='https://example.com/release',direction='bullish')])}
        s['events']['coverage']['complete']=False
        self.assertEqual(evaluate(47,s)['status'],'matched')
        s['events']['records'][0]['updated_at']='2026-11-09T18:00:00Z'
        self.assertEqual(evaluate(47,s)['status'],'missing_evidence')
        s['book']=envelope([dict(timestamp='2026-11-09T16:59:00Z',available_at='2026-11-09T16:59:00Z',sequence=1,type='snapshot',bids=[[100,100]],asks=[[101,10]])])
        s['book']['coverage']['complete']=False
        self.assertEqual(evaluate(43,s)['status'],'matched')
        self.assertFalse(evaluate(43,s)['evidence'][0]['coverage']['complete'])

    def test_cancellation_removes_original_trade_from_volume_profile(self):
        records=[dict(timestamp='2026-11-09T15:00:00Z',available_at='2026-11-09T15:00:00Z',price=100,size=1000,trade_id='one'),
                 dict(timestamp='2026-11-09T15:00:01Z',available_at='2026-11-09T15:00:01Z',price=102,size=10,trade_id='two'),
                 dict(timestamp='2026-11-09T15:00:02Z',available_at='2026-11-09T15:00:02Z',price=100,size=1000,trade_id='one',cancelled=True)]
        s={'cutoff':'2026-11-09T16:00:00Z','trades':envelope(records)}
        self.assertEqual(evaluate(38,s)['status'],'no_match')


    def test_gap_uses_previous_session_close_not_same_day_close(self):
        d=bars([100]*239+[105]);d.index=pd.bdate_range(end='2026-11-09',periods=240)
        s={'cutoff':'2026-11-09T16:00:00Z','intraday':envelope(intraday([102]*6+[103]))}
        self.assertEqual(evaluate(46,s,d)['direction'],'bullish')
        d.attrs['adjustment']='adjusted'
        self.assertEqual(evaluate(46,s,d)['status'],'missing_data')

    def test_xnys_weekend_rejected_and_early_close_profile_complete(self):
        s={'cutoff':'2026-11-07T18:00:00Z','intraday':envelope(intraday([100]*7,'2026-11-07'),session='2026-11-07')}
        self.assertEqual(evaluate(15,s)['status'],'invalid_data')
        records=intraday([100]*78,'2026-11-25')+intraday([102]*42,'2026-11-27')
        s={'cutoff':'2026-11-27T18:00:00Z','intraday':envelope(records,session='2026-11-27')}
        self.assertEqual(evaluate(44,s)['direction'],'bullish')

    def test_tape_can_describe_bounded_observed_coverage(self):
        records=[dict(timestamp='2026-11-09T15:00:00Z',available_at='2026-11-09T15:00:00Z',price=100,size=10) for _ in range(10)]
        s={'cutoff':'2026-11-09T16:00:00Z','trades':envelope(records),
           'quotes':envelope([dict(timestamp='2026-11-09T15:00:00Z',available_at='2026-11-09T15:00:00Z',bid=99,ask=100)])}
        s['trades']['coverage']={'complete':False,'status':'observed'}
        s['quotes']['coverage']={'complete':False,'status':'observed'}
        self.assertEqual(evaluate(42,s)['status'],'matched')
        self.assertFalse(evaluate(42,s)['evidence'][0]['coverage']['complete'])


    def test_nonfinite_metadata_abstains_and_serializes(self):
        s={'cutoff':'2026-11-09T17:00:00Z','intraday':envelope(intraday([100]*6+[102]))}
        s['intraday']['coverage']['fraction']=float('nan')
        row=evaluate(15,s)
        self.assertEqual(row['status'],'invalid_data')
        json.dumps(row,allow_nan=False)


    def test_alpaca_regular_sale_projects_to_observed_tape(self):
        from unittest.mock import patch
        from tradingagents.pilot.market_inputs import fetch_trades, fetch_quotes, for_symbol
        trade_page={'trades':{'ABC':[dict(t='2026-11-09T15:00:00Z',p=100,s=10,c=['@'],i=i) for i in range(10)]}}
        quote_page={'quotes':{'ABC':[dict(t='2026-11-09T15:00:00Z',bp=99,ap=100,bs=10,as_=10)]}}
        # Alpaca's ask-size wire key is "as".
        quote_page['quotes']['ABC'][0]['as']=10
        with patch('tradingagents.pilot.market_inputs.alpaca_get',return_value=trade_page):
            trades=fetch_trades(['ABC'],'2026-11-09T15:00:00Z','2026-11-09T15:01:00Z')
        with patch('tradingagents.pilot.market_inputs.alpaca_get',return_value=quote_page):
            quotes=fetch_quotes(['ABC'],'2026-11-09T15:00:00Z','2026-11-09T15:01:00Z')
        s={'symbol':'ABC','cutoff':'2026-11-09T15:01:00Z','trades':for_symbol(trades,'ABC'),'quotes':for_symbol(quotes,'ABC')}
        self.assertEqual(evaluate(42,s)['status'],'matched')
        self.assertEqual(evaluate(38,s)['status'],'missing_data')
        for r in s['trades']['records']:r['conditions']=['unknown']
        self.assertEqual(evaluate(42,s)['status'],'missing_data')


    def test_gap_unknown_adjustment_abstains_but_matching_prior_raw_reference_works(self):
        d=bars([100]*240);d.attrs.clear()
        s={'cutoff':'2026-11-09T16:00:00Z','intraday':envelope(intraday([102]*6+[103]))}
        self.assertEqual(evaluate(46,s,d)['status'],'missing_data')
        d.attrs['adjustment']='all'
        s['previous_close']={'price':100,'asof':'2026-11-06','adjustment':'raw','provider':'Alpaca','feed':'sip'}
        row=evaluate(46,s,d)
        self.assertEqual(row['direction'],'bullish')
        self.assertEqual(row['evidence'][0]['previous_close_time'],'2026-11-06')
        s['previous_close']['adjustment']='all'
        self.assertEqual(evaluate(46,s,d)['status'],'missing_data')
        s['previous_close'].update(adjustment='raw',asof='2026-11-09')
        self.assertEqual(evaluate(46,s,d)['status'],'invalid_data')

    def test_supplemental_matches_serialize_strictly(self):
        s={'cutoff':'2026-11-09T17:00:00Z','intraday':envelope(intraday([100]*6+[102]))}
        json.dumps(evaluate_stock(bars([100]*240),supplemental=s),allow_nan=False)


if __name__=='__main__':
    unittest.main()
