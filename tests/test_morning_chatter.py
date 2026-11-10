import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from tradingagents.pilot.morning import collect, top_tickers

class MorningTests(unittest.TestCase):
    @patch('tradingagents.pilot.morning.news')
    def test_batches_dedup_and_union(self,news):
        news.side_effect=[([{'id':'same','symbols':['A'],'kind':'news'}],{'truncated':False}),([{'id':'same','symbols':['B'],'kind':'news'}],{'truncated':False})]
        rows,status=collect([str(i) for i in range(51)],'2026-11-09T13:00:00Z','2026-11-08T13:00:00Z')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['symbols'],['A','B'])
        self.assertEqual(status['successful_symbols'],51);self.assertEqual(news.call_count,2)
    @patch('tradingagents.pilot.morning.news')
    def test_failure_does_not_skip_later_batches(self,news):
        news.side_effect=[ValueError('HTTP 403'),([] ,{'truncated':True})]
        _,status=collect([str(i) for i in range(51)],'2026-11-09T13:00:00Z','2026-11-08T13:00:00Z')
        self.assertEqual(status['status'],'partial');self.assertEqual(status['successful_symbols'],1)
        self.assertEqual(len(status['errors']),1);self.assertEqual(len(status['truncated_batches']),1)
    def test_rank_order_not_file_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pd.DataFrame({'rank':range(1001,0,-1),'ticker':[str(x) for x in range(1001,0,-1)]}).to_csv(p/'universe.csv',index=False)
            tickers=top_tickers(p);self.assertEqual(len(tickers),1000);self.assertEqual(int(tickers[0]),1)

if __name__=='__main__':unittest.main()
