import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from tradingagents.pilot.screen import build_row,spread_assessment,market_clock,identity_check
from tradingagents.pilot.providers import alpaca_daily_bars,alpaca_get

class PilotTests(unittest.TestCase):
    def bars(self,volume=1e6):
        import exchange_calendars as x
        dates=x.get_calendar('XNYS').sessions_in_range('2025-10-01','2026-11-06')
        c=np.linspace(20,30,len(dates))
        return pd.DataFrame(dict(Open=c,High=c+.5,Low=c-.5,Close=c,Volume=volume),index=dates)
    def member(self):return dict(ticker='A',name='A Common Stock',rank=1)
    def test_session_cutoff_market_open_and_early_close(self):
        self.assertEqual(market_clock('2026-11-09T15:00:00Z')['asof'],'2026-11-06')
        self.assertTrue(market_clock('2026-11-09T15:00:00Z')['regular_session_open'])
        self.assertEqual(market_clock('2026-11-27T18:10:00Z')['asof'],'2026-11-25')
        self.assertEqual(market_clock('2026-11-27T18:16:00Z')['asof'],'2026-11-27')
    def test_median_resists_single_spike(self):
        d=self.bars(100000);d.iloc[-1,d.columns.get_loc('Volume')]=1e9
        r=build_row(self.member(),d,d,'2026-11-06')
        self.assertEqual(r['liquidity_tier'],'thin');self.assertEqual(r['eligibility'],'excluded')
    def test_liquidity_secondary_and_preferred(self):
        for volume,tier in [(300000,'secondary'),(1e6,'preferred')]:
            d=self.bars(volume);r=build_row(self.member(),d,d,'2026-11-06')
            self.assertEqual(r['liquidity_tier'],tier);self.assertEqual(r['eligibility'],'provisional')
    def test_missing_session_and_stale(self):
        d=self.bars()
        for broken in (d.iloc[:-1],d.drop(d.index[-3])):
            r=build_row(self.member(),broken,d,'2026-11-06')
            self.assertEqual(r['eligibility'],'data_review')
    def test_incomplete_today_excluded(self):
        d=self.bars();future=d.iloc[[-1]].copy();future.index=pd.DatetimeIndex(['2026-11-09']);future['Close']=10000
        r=build_row(self.member(),pd.concat([d,future]),d,'2026-11-06')
        self.assertEqual(r['close'],30)
    def test_instrument_identity(self):
        self.assertEqual(identity_check({'ticker':'A','name':'A preferred stock'}),'excluded_security_type')
        self.assertEqual(identity_check(self.member(),{'symbol':'B','quote_type':'EQUITY','exchange':'NASDAQ'}),'symbol_mismatch')
        self.assertEqual(identity_check(self.member(),{'symbol':'A','quote_type':'EQUITY','exchange':'NASDAQ'}),'provider_verified_equity')
    def samples(self):
        return [dict(provider='Alpaca',feed='iex',bid=100,ask=100.1,quote_at=f'2026-11-09T15:00:0{i}Z',observed_at=f'2026-11-09T15:00:0{i}Z',regular_session=True) for i in range(3)]
    def test_spread_repeated_cached_sample_not_three(self):
        s=self.samples()[0];r=spread_assessment([s,s,s],'2026-11-09T15:00:04Z')
        self.assertEqual(r['status'],'sampling');self.assertEqual(r['samples'],1)
    def test_spread_valid_and_wide(self):
        s=self.samples();self.assertEqual(spread_assessment(s,'2026-11-09T15:00:04Z')['status'],'preferred')
        for q in s:q['ask']=101
        self.assertEqual(spread_assessment(s,'2026-11-09T15:00:04Z')['status'],'wide')
    def test_spread_missing_stale_crossed_afterhours(self):
        for change in ({'quote_at':None},{'bid':101},{'regular_session':False},{'quote_at':'2026-11-09T14:00:00Z'}):
            s=[dict(q,**change) for q in self.samples()]
            self.assertEqual(spread_assessment(s,'2026-11-09T15:00:04Z')['status'],'unchecked')
        self.assertEqual(spread_assessment(self.samples(),'2026-11-09T15:05:00Z')['status'],'unchecked')
    @patch('tradingagents.pilot.providers.alpaca_get')
    def test_alpaca_pagination_feed_and_adjustment(self,get):
        get.side_effect=[dict(bars={'A':[dict(t='2026-11-05T05:00:00Z',o=20,h=21,l=19,c=20,v=100)]},next_page_token='next'),dict(bars={'B':[dict(t='2026-11-06T05:00:00Z',o=30,h=31,l=29,c=30,v=200)]},next_page_token=None)]
        data=alpaca_daily_bars(['A','B'],'2026-11-06')
        self.assertEqual(set(data),{'A','B'})
        self.assertEqual(str(data['B'].index[-1].date()),'2026-11-06')
        for call in get.call_args_list:
            self.assertEqual(call.args[1]['feed'],'sip');self.assertEqual(call.args[1]['adjustment'],'all')
        self.assertEqual(get.call_args_list[-1].args[1]['page_token'],'next')
    def test_no_order_endpoint(self):
        with self.assertRaises(ValueError):alpaca_get('/v2/orders')
    @patch('tradingagents.pilot.providers.alpaca_headers',return_value={})
    @patch('tradingagents.pilot.providers.requests.get')
    def test_permission_failure_never_falls_back_to_iex(self,get,headers):
        get.return_value.status_code=403
        with self.assertRaisesRegex(ValueError,'No automatic feed substitution'):alpaca_daily_bars(['A'],'2026-11-06')
        self.assertEqual(get.call_count,1)

if __name__=='__main__':unittest.main()
