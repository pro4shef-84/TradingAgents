"""Offline timing, geometry, validation and execution-price regression tests."""

import unittest

import numpy as np
import pandas as pd

from tradingagents.technical_scan import evaluate_events, indicators, pivots, scan, validate_bars


def bars(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + .2, "Low": close - .2,
                         "Close": close, "Volume": np.full(len(close), 1000.)},
                        index=pd.bdate_range("2023-01-02", periods=len(close)))


class TechnicalScanTests(unittest.TestCase):
    def test_flat_market_is_not_a_signal(self):
        result = scan(bars(np.full(240, 100.)))
        self.assertEqual(result["signals"], [])
        self.assertEqual(result["context"]["rsi"], 50)

    def test_breakout_uses_prior_range(self):
        df = bars([100.] * 220 + [105.])
        df.iloc[-1, df.columns.get_loc("Volume")] = 2000
        result = scan(df)
        breakout = [s for s in result["signals"] if s["strategy"] == "volume_breakout"]
        self.assertEqual(len(breakout), 1)
        self.assertEqual(breakout[0]["level"], 100.2)
        self.assertEqual(breakout[0]["evidence"]["relative_volume"], 2)

    def test_cutoff_invariance(self):
        df = bars(np.r_[np.linspace(80, 110, 220), 115, 1, 900])
        cutoff = df.index[-3]
        self.assertEqual(scan(df, asof=cutoff), scan(df.loc[:cutoff]))

    def test_pivot_confirmation(self):
        x = np.array([1, 2, 3, 9, 3, 2, 1])
        self.assertEqual(pivots(x[:-1]), [])
        self.assertEqual(pivots(x), [3])

    def test_double_top_and_inverse(self):
        values = np.interp(np.arange(80), [0, 30, 40, 50, 60, 69, 78, 79],
                           [80, 95, 110, 100, 110, 104, 101, 99])
        for close, expected in [(values, "Double top"), (200 - values, "Double bottom")]:
            df = bars(close)
            result = scan(df)
            hits = [s for s in result["signals"] if s["strategy"] == "double_top_bottom"]
            self.assertEqual([s["variant"] for s in hits], [expected])
            self.assertFalse(any(s["strategy"] == "double_top_bottom" for s in scan(df.iloc[:-1])["signals"]))

    def test_head_shoulders_and_inverse(self):
        values = np.interp(np.arange(100), [0, 30, 40, 50, 60, 70, 80, 90, 98, 99],
                           [80, 95, 110, 100, 118, 100, 110, 104, 101, 99])
        for close, expected in [(values, "Head and shoulders"), (220 - values, "Inverse head and shoulders")]:
            hits = [s for s in scan(bars(close))["signals"] if s["strategy"] == "head_shoulders"]
            self.assertEqual([s["variant"] for s in hits], [expected])

    def test_triangle_breakout_and_mirror(self):
        values = np.interp(np.arange(60), [0, 10, 15, 25, 30, 40, 45, 58, 59],
                           [100, 110, 100, 110, 103, 110, 106, 109, 112])
        for close, expected in [(values, "Ascending triangle"), (220 - values, "Descending triangle")]:
            hits = [s for s in scan(bars(close))["signals"] if s["strategy"] == "triangle_wedge"]
            self.assertEqual([s["variant"] for s in hits], [expected])

    def test_insufficient_history_and_bad_data(self):
        with self.assertRaises(ValueError):
            scan(bars([100.] * 59))
        df = bars([100.] * 220)
        df.iloc[-1, 1] = 99
        with self.assertRaises(ValueError):
            scan(df)
        df = bars([100.] * 220)
        df.iloc[-1, 4] = np.nan
        with self.assertRaises(ValueError):
            scan(df)
        with self.assertRaises(ValueError):
            validate_bars(pd.concat([df.iloc[:1], df.iloc[:1]]))

    def test_warmup_is_explicit(self):
        result = scan(bars(np.linspace(90, 100, 90)))
        self.assertIsNone(result["context"]["sma200"])
        self.assertTrue(result["coverage_notes"])

    def test_ohlc_float_roundtrip_tolerance_is_not_price_tolerance(self):
        df = bars([100.] * 220)
        df.iloc[-1, df.columns.get_loc("High")] = np.nextafter(100., 0.)
        validate_bars(df)
        df.iloc[-1, df.columns.get_loc("High")] = 99.999999
        with self.assertRaises(ValueError):
            validate_bars(df)

    def test_rsi_extremes(self):
        self.assertEqual(indicators(bars(np.arange(1, 101)))["rsi"].iloc[-1], 100)
        self.assertEqual(indicators(bars(np.arange(100, 0, -1)))["rsi"].iloc[-1], 0)

    def test_evaluation_enters_next_open_and_subtracts_costs(self):
        df = bars([100.] * 220 + [105., 110., 121.])
        result = evaluate_events(df, holding_bars=2, cost_bps=20, start=df.index[220])
        self.assertTrue(result["events"])
        for event in result["events"]:
            self.assertEqual(event["entry"], 110.)
            self.assertEqual(event["exit"], 121.)
            self.assertAlmostEqual(event["net_return"], .098)
            self.assertGreater(event["entry_date"], event["date"])

    def test_zero_volume_serializable(self):
        import json

        df = bars(np.full(220, 100.))
        df["Volume"] = 0
        json.dumps(scan(df), allow_nan=False)


    def test_bearish_event_return_and_invalid_evaluation_parameters(self):
        df = bars([100.] * 220 + [95., 90., 81.])
        result = evaluate_events(df, holding_bars=2, cost_bps=20, start=df.index[220])
        self.assertTrue(result["events"])
        for event in result["events"]:
            self.assertEqual(event["direction"], "bearish")
            self.assertAlmostEqual(event["net_return"], .098)
        for hold, cost in [(0, 10), (10, -1), (10, float("nan"))]:
            with self.assertRaises(ValueError):
                evaluate_events(df, hold, cost)


if __name__ == "__main__":
    unittest.main()
