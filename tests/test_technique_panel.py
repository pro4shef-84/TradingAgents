"""Specialist coverage, temporal causality and family-capped judgment tests."""

import copy
import json
import unittest

import numpy as np
import pandas as pd

from tradingagents.technique_panel import CATALOG, evaluate_stock, judge, rank_agent_rows, research_sort_key


def bars(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + .5, "Low": close - .5,
                         "Close": close, "Volume": np.full(len(close), 1_000_000.)},
                        index=pd.bdate_range("2023-01-02", periods=len(close)))


class PanelTests(unittest.TestCase):
    def test_registry_uses_original_fifty_numbers(self):
        self.assertEqual([r["id"] for r in CATALOG], list(range(1, 51)))
        self.assertEqual(CATALOG[40]["name"], "Volume climax")
        self.assertEqual(CATALOG[41]["name"], "Tape reading")

    def test_all_fifty_report_with_explicit_abstentions(self):
        panel = evaluate_stock(bars([100] * 240))
        self.assertEqual(len(panel["agents"]), 50)
        by_id = {r["id"]: r for r in panel["agents"]}
        self.assertEqual(by_id[15]["status"], "missing_data")
        self.assertEqual(by_id[25]["status"], "no_match")
        self.assertIn("Price-zone", by_id[25]["name"])
        self.assertEqual(by_id[47]["status"], "missing_evidence")
        self.assertEqual(by_id[48]["status"], "missing_data")
        self.assertIsNone(by_id[48]["fit_score"])
        self.assertFalse(any(r["status"] == "matched" for r in panel["agents"]))
        json.dumps(panel, allow_nan=False)

    def test_future_bars_do_not_change_results(self):
        df = bars(np.r_[np.linspace(80, 110, 240), 120, 1, 900])
        benchmark = bars(np.linspace(90, 100, len(df)))
        cutoff = df.index[-3]
        self.assertEqual(evaluate_stock(df, benchmark, cutoff), evaluate_stock(df.loc[:cutoff], benchmark.loc[:cutoff]))

    def test_relative_strength_benchmark_alignment(self):
        df = bars(np.linspace(50, 150, 240))
        bench = bars([100] * 240)
        row = evaluate_stock(df, bench)["agents"][47]
        self.assertEqual((row["status"], row["direction"]), ("matched", "bullish"))
        row = evaluate_stock(df, bench.iloc[:-1])["agents"][47]
        self.assertEqual(row["status"], "missing_data")

    def test_family_votes_do_not_multiply(self):
        df = bars(np.linspace(80, 110, 240))
        panel = evaluate_stock(df)
        fake = {"status": "matched", "family": "momentum", "fit_score": 80,
                "events": [{"direction": "bullish"}]}
        panel["agents"] = [copy.deepcopy(fake) for _ in range(50)]
        result = judge(panel, df)
        self.assertEqual(result["supporting_family_count"], 1)
        self.assertEqual(result["decision"], "research_priority")

    def test_watch_and_context_are_not_confirmed_votes(self):
        df = bars(np.linspace(80, 110, 240))
        panel = evaluate_stock(df)
        panel["agents"] = [{"status": "watch", "family": "structure", "fit_score": 40,
                            "events": [{"direction": "bullish"}]}]
        self.assertEqual(judge(panel, df)["supporting_family_count"], 0)

    def test_ranks_separate_direction_and_exclude_missing(self):
        rows = [{"id": 1, "status": "matched", "direction": "bullish", "fit_score": 80},
                {"id": 1, "status": "matched", "direction": "bearish", "fit_score": 70},
                {"id": 1, "status": "matched", "direction": "bullish", "fit_score": 80},
                {"id": 1, "status": "no_match", "direction": "neutral", "fit_score": 0},
                {"id": 1, "status": "missing_data", "direction": None, "fit_score": None}]
        self.assertEqual([r["agent_rank"] for r in rank_agent_rows(rows)], [1, 1, 1, None, None])

    def test_insufficient_history_never_becomes_negative_vote(self):
        df = bars([100] * 30)
        panel = evaluate_stock(df)
        self.assertTrue(all(r["status"] == "insufficient_history" for r in panel["agents"]))
        self.assertEqual(judge(panel, df)["decision"], "insufficient_data")

    def test_nearby_resistance_is_context_only(self):
        df = bars(np.linspace(80, 100, 240))
        df.iloc[-4, df.columns.get_loc("High")] = 101.
        panel = evaluate_stock(df)
        panel["agents"] = [
            {"status": "matched", "family": family, "fit_score": 90,
             "events": [{"direction": "bullish"}]} for family in ("structure", "volume")]
        result = judge(panel, df)
        self.assertEqual(result["nearest_resistance"], 101.)
        self.assertLess(result["net_reward_risk_10bps"], 2)
        self.assertEqual(result["decision"], "research_priority")

    def test_inside_bar_is_watch_only(self):
        df = bars([100] * 240)
        df.iloc[-1, df.columns.get_loc("High")] = 100.2
        df.iloc[-1, df.columns.get_loc("Low")] = 99.8
        row = evaluate_stock(df)["agents"][17]
        self.assertEqual((row["status"], row["direction"]), ("watch", "neutral"))

    def test_flag_requires_impulse_and_breakout_both_directions(self):
        close = np.r_[[100.] * 220, np.linspace(100, 120, 11), np.linspace(119.8, 118.4, 7), 121.5]
        for values, direction in ((close, "bullish"), (240 - close, "bearish")):
            row = evaluate_stock(bars(values))["agents"][6]
            self.assertEqual((row["status"], row["direction"]), ("matched", direction))
            df = bars(values)
            df.iloc[-1] = df.iloc[-2]
            self.assertEqual(evaluate_stock(df)["agents"][6]["status"], "no_match")

    def test_sar_flip_after_uptrend(self):
        df = bars(np.r_[np.linspace(80, 120, 239), 100])
        row = evaluate_stock(df)["agents"][33]
        self.assertEqual((row["status"], row["direction"]), ("matched", "bearish"))

    def test_ichimoku_has_explicit_full_cloud_warmup(self):
        df = bars(np.linspace(80, 100, 65))
        self.assertEqual(evaluate_stock(df)["agents"][32]["status"], "insufficient_history")

    def test_judge_can_pass_two_independent_families_with_real_room(self):
        df = bars(np.linspace(80, 100, 240))
        df.iloc[-20, df.columns.get_loc("High")] = 110.
        panel = evaluate_stock(df)
        panel["agents"] = [{"status": "matched", "family": family, "fit_score": 90,
                            "events": [{"direction": "bullish"}]} for family in ("structure", "volume")]
        result = judge(panel, df)
        self.assertEqual(result["decision"], "research_priority")
        self.assertGreaterEqual(result["net_reward_risk_10bps"], 2)
        panel["agents"].append({"status": "matched", "family": "momentum", "fit_score": 70,
                                "events": [{"direction": "bearish"}]})
        self.assertEqual(judge(panel, df)["decision"], "research_priority")

    def test_ranking_ignores_risk_and_conflict_vetoes(self):
        row = {"decision": "research_priority", "bullish_families": {"trend": 80},
               "bearish_families": {}, "market_cap_rank": 1}
        modified = {**row, "net_reward_risk_10bps": -100, "entry": 1,
                    "reasons": ["old veto"], "opposing_family_count": 20}
        self.assertEqual(research_sort_key(row), research_sort_key(modified))

    def test_low_price_liquidity_and_short_history_do_not_veto(self):
        df = bars(np.linspace(2, 4, 80))
        df["Volume"] = 100
        panel = evaluate_stock(df)
        panel["agents"] = [{"status": "matched", "family": "momentum", "fit_score": 80,
                            "events": [{"direction": "bullish"}]}]
        result = judge(panel, df)
        self.assertEqual(result["decision"], "research_priority")
        self.assertTrue(result["risk_context_only"])
        self.assertIsNone(result["nearest_resistance"])

    def test_bearish_and_mixed_evidence_are_not_blocked(self):
        df = bars(np.linspace(100, 80, 240))
        panel = evaluate_stock(df)
        panel["agents"] = [{"status": "matched", "family": "momentum", "fit_score": 80,
                            "events": [{"direction": "bearish"}]}]
        result = judge(panel, df)
        self.assertEqual(result["decision"], "research_priority")
        self.assertEqual(result["evidence_direction"], "bearish")
        panel["agents"][0]["events"].append({"direction": "bullish"})
        self.assertEqual(judge(panel, df)["evidence_direction"], "mixed")
        self.assertEqual(judge(panel, df)["decision"], "research_priority")


if __name__ == "__main__":
    unittest.main()
