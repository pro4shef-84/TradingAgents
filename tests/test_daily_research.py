import json
import tempfile
import unittest
from pathlib import Path


def row(ticker='A', **updates):
    value = dict(ticker=ticker, name='Example', market_cap_rank=1, asof='2026-11-06',
                 close=100., eligibility='eligible', liquidity_tier='preferred',
                 median_dollar_volume=30e6, research_queue='continuation',
                 assessment=dict(decision='research_priority', bullish_families={'trend':80},
                                 bearish_families={}, evidence_direction='bullish'),
                 agents=[dict(id=49,name='Breakout',status='matched',direction='bullish',
                              family='trend',events=[dict(direction='bullish',level=99.,evidence='Break')])],
                 context={}, source='fixture')
    return {**value, **updates}


class PipelineTests(unittest.TestCase):
    def test_bad_quote_symbol_preserves_other_symbols_and_reports_gap(self):
        import pandas as pd
        from unittest.mock import patch
        from tradingagents.pilot.daily import _enrich
        from tradingagents.pilot import market_inputs
        from tradingagents import technique_panel

        def sample(symbols, *args, **kwargs):
            symbol=symbols[0]
            return dict(records={symbol:[]},coverage={symbol:{'status':'observed','complete':False}})

        def quotes(symbols, *args, **kwargs):
            if symbols==['BAD']:raise ValueError('Crossed quote')
            return sample(symbols)

        observed={}
        def evaluate(bars, asof, supplemental):
            observed[supplemental['symbol']]=supplemental
            return {'agents':[]}

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame({'Close':[100.]},index=pd.to_datetime(['2026-11-06'])).to_csv(root/'prices.csv')
            rows=[row(t,price_file='prices.csv') for t in ('A','BAD','C')]
            with patch.object(market_inputs,'fetch_intraday',side_effect=ValueError('No intraday')), \
                 patch.object(market_inputs,'fetch_trades',side_effect=sample), \
                 patch.object(market_inputs,'fetch_quotes',side_effect=quotes), \
                 patch.object(technique_panel,'evaluate_stock',side_effect=evaluate), \
                 patch.object(technique_panel,'judge',return_value={'evidence_direction':'bullish'}):
                result=_enrich(root,rows,'2026-11-09T13:00:00Z','2026-11-06',root)
            saved=json.loads((root/'supplemental.json').read_text())
            self.assertEqual(set(saved['quotes']['records']),{'A','C'})
            self.assertEqual(result['errors']['quotes'],{'BAD':'Crossed quote'})
            self.assertEqual(result['status'],'partial')
            self.assertEqual(observed['BAD']['quotes']['coverage']['status'],'missing')
            self.assertEqual(observed['C']['quotes']['coverage']['status'],'observed')


    def test_queue_recomputed_after_new_directional_evidence(self):
        from tradingagents.pilot.screen import research_queue
        self.assertEqual(research_queue({'evidence_direction':'mixed'}, {}, [], 'eligible'),'conflicting')
        self.assertEqual(research_queue({'evidence_direction':'bearish'}, {}, [], 'eligible'),'bearish')
        self.assertEqual(research_queue({'evidence_direction':'bullish'}, {'regime':'uptrend'}, [], 'excluded'),'excluded')
