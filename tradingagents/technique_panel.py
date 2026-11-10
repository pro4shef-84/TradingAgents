"""Fifty named specialist evaluators and an evidence-based research judge.

These are deterministic specialist modules, not fifty independent LLM opinions.
Scores measure rule fit, never likelihood of profit. Unsupported rules and data
are explicit abstentions. All evaluation uses completed bars through the cutoff.
"""

from collections import defaultdict

from tradingagents.specialist_rules import daily_rules, supplemental_rules

import numpy as np
import pandas as pd

from tradingagents.technical_scan import indicators, pivots, scan, validate_bars

# Original user-supplied numbering, including 41=climax, 42=tape, 43=Level 2.
_SPEC = [
    (1, "Head and shoulders / inverse", "structure", "head_shoulders"),
    (2, "Double top / bottom", "structure", "double_top_bottom"),
    (3, "Triple top / bottom", "structure", "triple"),
    (4, "Rising / falling wedge", "structure", "wedge"),
    (5, "Rounding bottom", "structure", "rounding_bottom"),
    (6, "Diamond top / bottom", "structure", "diamond"),
    (7, "Bull / bear flag", "structure", "flag"),
    (8, "Bull / bear pennant", "structure", "pennant"),
    (9, "Ascending triangle", "structure", "ascending"),
    (10, "Descending triangle", "structure", "descending"),
    (11, "Symmetrical triangle", "structure", "symmetrical"),
    (12, "Cup and handle", "structure", "cup_handle"),
    (13, "Rectangle breakout", "structure", "rectangle"),
    (14, "Measuring gap", "structure", "measuring_gap"),
    (15, "Opening range breakout", "intraday", "intraday"),
    (16, "Hammer / shooting star", "candles", "hammer_star"),
    (17, "Bullish / bearish engulfing", "candles", "engulfing"),
    (18, "Inside bar", "candles", "inside"),
    (19, "Marubozu", "candles", "marubozu"),
    (20, "Morning / evening star", "candles", "star"),
    (21, "Three soldiers / crows", "candles", "three"),
    (22, "Doji variants", "candles", "doji"),
    (23, "Tweezer top / bottom", "candles", "tweezer"),
    (24, "Harami cross", "candles", "harami"),
    (25, "Price-zone reaction proxy (historical: Order blocks)", "structure", "price_zone"),
    (26, "Moving-average crossovers", "trend", "ma"),
    (27, "RSI divergence", "momentum", "rsi_divergence"),
    (28, "MACD histogram expansion", "momentum", "macd_histogram"),
    (29, "Bollinger squeeze breakout", "volatility", "squeeze_breakout"),
    (30, "Session VWAP", "intraday", "intraday"),
    (31, "Stochastic recovery", "momentum", "stochastic"),
    (32, "ATR expansion", "volatility", "atr"),
    (33, "Ichimoku cloud breakout", "trend", "ichimoku"),
    (34, "Parabolic SAR flip", "trend", "sar"),
    (35, "Fibonacci retracement", "structure", "fibonacci"),
    (36, "Prior-session pivot points", "structure", "pivot_points"),
    (37, "OBV breakout", "volume", "obv"),
    (38, "Volume profile POC", "intraday", "intraday_volume"),
    (39, "Wyckoff accumulation / distribution", "structure", "wyckoff"),
    (40, "Failed break / reclaim (sweep proxy)", "structure", "sweep"),
    (41, "Volume climax", "volume", "climax"),
    (42, "Tape reading", "microstructure", "trades"),
    (43, "Level 2 imbalance", "microstructure", "order_book"),
    (44, "Market Profile value-area shifts", "intraday", "intraday"),
    (45, "Mean-reversion recovery", "momentum", "mean_reversion"),
    (46, "Gap and go", "intraday", "intraday"),
    (47, "Catalyst momentum", "catalyst", "catalyst"),
    (48, "Relative strength outperformance", "relative_strength", "relative_strength"),
    (49, "20/55-bar trend breakout", "trend", "donchian_breakout"),
    (50, "Pairs trading", "relative_value", "paired_data"),
]
CATALOG = [{"id": i, "name": name, "family": family, "rule": rule}
           for i, name, family, rule in _SPEC]
ABSTENTIONS = {
    "undefined": ("rule_not_implemented", "Requires a separately specified and tested pattern rule; no result inferred"),
    "unobservable": ("not_supported", "Daily candles cannot identify institutional owners or unfilled orders"),
    "intraday": ("missing_data", "Requires completed intraday bars with session boundaries"),
    "intraday_volume": ("missing_data", "Requires intraday volume-at-price observations"),
    "trades": ("missing_data", "Requires timestamped trade prints"),
    "order_book": ("missing_data", "Requires order-book depth and updates"),
    "paired_data": ("missing_data", "Requires a defined pair, tested spread relationship and short-cost assumptions"),
    "catalyst": ("missing_evidence", "Requires timestamped company/catalyst evidence; price bars alone are insufficient"),
}


def _sar(d):
    """Wilder-style SAR with 0.02 acceleration, capped at 0.2; two-bar clamp."""
    h, low, c = d.High.to_numpy(), d.Low.to_numpy(), d.Close.to_numpy()
    out = np.full(len(d), np.nan)
    bull = c[1] >= c[0]
    value = min(low[:2]) if bull else max(h[:2])
    extreme = max(h[:2]) if bull else min(low[:2])
    acceleration = .02
    out[1] = value
    directions = [bull, bull]
    for i in range(2, len(d)):
        value += acceleration * (extreme - value)
        if bull:
            value = min(value, low[i - 1], low[i - 2])
            if low[i] < value:
                bull, value, extreme, acceleration = False, extreme, low[i], .02
            elif h[i] > extreme:
                extreme, acceleration = h[i], min(.2, acceleration + .02)
        else:
            value = max(value, h[i - 1], h[i - 2])
            if h[i] > value:
                bull, value, extreme, acceleration = True, extreme, h[i], .02
            elif low[i] < extreme:
                extreme, acceleration = low[i], min(.2, acceleration + .02)
        out[i] = value
        directions.append(bull)
    return out, directions


def _extra_events(d, benchmark):
    events = defaultdict(list)
    now, previous = d.iloc[-1], d.iloc[-2]
    atr, price = float(now.atr), float(now.Close)
    if atr <= 0:
        return events

    def add(rule, direction, evidence, level=None, status="matched"):
        events[rule].append({"direction": direction, "evidence": evidence,
                             "level": price if level is None else float(level), "status": status})

    for sign, direction in ((1, "bullish"), (-1, "bearish")):
        x = sign * d.Close.to_numpy()
        upper = d.High.to_numpy() if sign == 1 else -d.Low.to_numpy()
        lower = d.Low.to_numpy() if sign == 1 else -d.High.to_numpy()
        # Fresh break of a 20-bar box with >=2 touches of each boundary.
        top, bottom = max(upper[-21:-1]), min(lower[-21:-1])
        if (x[-1] > top >= x[-2] and top - bottom <= 6 * atr
                and abs(x[-2] - x[-21]) <= .5 * (top - bottom)
                and sum(abs(upper[-21:-1] - top) <= .3 * atr) >= 2
                and sum(abs(lower[-21:-1] - bottom) <= .3 * atr) >= 2):
            add("rectangle", direction, "20-bar range <=6 ATR; two boundary touches each; fresh close breakout", sign * top)
        # Eight-bar continuation after a ten-bar impulse; explicit heuristic.
        impulse = x[-9] - x[-19]
        hs, _ = np.polyfit(np.arange(8), upper[-9:-1], 1)
        ls, _ = np.polyfit(np.arange(8), lower[-9:-1], 1)
        flag_top, flag_bottom = max(upper[-9:-1]), min(lower[-9:-1])
        if (impulse >= 4 * atr and flag_top - flag_bottom < .5 * impulse
                and x[-1] > flag_top >= x[-2] and x[-2] > x[-9] - .5 * impulse):
            if hs < 0 and ls < 0 and abs(hs - ls) <= .1 * atr:
                add("flag", direction, "10-bar impulse >=4 ATR; 8-bar countertrend channel; fresh breakout", sign * flag_top)
            elif hs < 0 < ls:
                add("pennant", direction, "10-bar impulse >=4 ATR; 8-bar converging consolidation; fresh breakout", sign * flag_top)
        prior_trend = sign * (d.Close.iloc[-2] - d.Close.iloc[-7]) < -atr
        body = abs(now.Close - now.Open)
        width = now.High - now.Low
        prior_body = abs(previous.Close - previous.Open)
        if width > 0 and body / width >= .9 and width >= atr and sign * (now.Close - now.Open) > 0:
            add("marubozu", direction, "Body >=90% of range; range >=ATR")
        first, middle = d.iloc[-3], d.iloc[-2]
        if (sign * (d.Close.iloc[-4] - d.Close.iloc[-9]) < -atr
                and sign * (first.Close - first.Open) < 0
                and abs(first.Close - first.Open) >= .5 * atr
                and abs(middle.Close - middle.Open) <= .3 * abs(first.Close - first.Open)
                and sign * (now.Close - now.Open) > 0
                and sign * (now.Close - (first.Open + first.Close) / 2) > 0):
            add("star", direction, "Three-candle reversal, middle body <=30% of first; midpoint recovered; gaps not required")
        last = d.iloc[-3:]
        if (sign * (d.Close.iloc[-4] - d.Close.iloc[-9]) < -atr
                and (sign * (last.Close - last.Open) > .5 * atr).all()
                and (sign * last.Close.diff().dropna() > 0).all()
                and all(min(d.Open.iloc[i - 1], d.Close.iloc[i - 1]) <= d.Open.iloc[i]
                        <= max(d.Open.iloc[i - 1], d.Close.iloc[i - 1]) for i in (-2, -1))):
            add("three", direction, "Three advancing bodies >0.5 ATR; opens inside prior bodies; preceding opposite trend")
        extreme = now.Low if sign == 1 else now.High
        prior_extreme = previous.Low if sign == 1 else previous.High
        if prior_trend and abs(extreme - prior_extreme) <= .1 * atr and sign * (now.Close - now.Open) > 0 and sign * (previous.Close - previous.Open) < 0:
            add("tweezer", direction, "Two extremes within 0.1 ATR; opposite-colour reversal after preceding trend", extreme)
        if (prior_trend and width > 0 and body <= .1 * width and prior_body >= .5 * atr
                and min(previous.Open, previous.Close) <= now.Low <= now.High <= max(previous.Open, previous.Close)):
            add("harami", direction, "Doji range inside preceding large body after opposite trend", status="watch")
        # Prior 20-bar extreme breached intrabar and reclaimed on the close.
        boundary = min(lower[-21:-1])
        if lower[-1] < boundary and x[-1] > boundary and x[-2] >= boundary:
            add("sweep", direction, "Failed break and reclaim of prior 20-bar extreme; order intent unknown", sign * boundary)
        # Three confirmed close pivots, approximately level peaks and troughs.
        z = -sign * d.Close.to_numpy()
        peaks = [i for i in pivots(z) if i >= len(d) - 120]
        if len(peaks) >= 3:
            p, q, r = peaks[-3:]
            v1 = p + int(np.argmin(z[p:q + 1]))
            v2 = q + int(np.argmin(z[q:r + 1]))
            neckline = min(z[v1], z[v2])
            if (min(q - p, r - q) >= 6 and r - p >= 20
                    and max(z[[p, q, r]]) - min(z[[p, q, r]]) <= atr
                    and abs(z[v1] - z[v2]) <= atr
                    and min(z[[p, q, r]]) - max(z[v1], z[v2]) >= atr
                    and p >= 10 and z[p] - min(z[max(0, p - 20):p]) >= 2 * atr
                    and min(z[r + 1:-1]) >= neckline and z[-1] < neckline):
                add("triple", direction, "Three confirmed close pivots within 1 ATR; two troughs within 1 ATR; fresh neckline break", -sign * neckline)
        # RSI divergence is only a watch condition, available after pivot confirmation.
        peaks = [i for i in pivots(-sign * d.Close.to_numpy()) if i >= len(d) - 60]
        if len(peaks) >= 2:
            p, q = peaks[-2:]
            if (q == len(d) - 4 and q - p >= 5 and sign * (d.Close.iloc[q] - d.Close.iloc[p]) < 0
                    and sign * (d.rsi.iloc[q] - d.rsi.iloc[p]) >= 5):
                add("rsi_divergence", direction, "Price makes new extreme while RSI diverges >=5 points; pivot confirmed 3 bars later", status="watch")
        hist = d.macd - d.macd_signal
        if sign * hist.iloc[-1] > 0 >= sign * hist.iloc[-2]:
            add("macd_histogram", direction, "MACD histogram crosses zero; equivalent to signal-line crossover")
        cloud9 = (d.High.rolling(9).max() + d.Low.rolling(9).min()) / 2
        cloud26 = (d.High.rolling(26).max() + d.Low.rolling(26).min()) / 2
        span_a = ((cloud9 + cloud26) / 2).shift(26)
        span_b = ((d.High.rolling(52).max() + d.Low.rolling(52).min()) / 2).shift(26)
        edge = pd.concat([span_a, span_b], axis=1).max(axis=1) if sign == 1 else pd.concat([span_a, span_b], axis=1).min(axis=1)
        if len(d) >= 79 and sign * (price - edge.iloc[-1]) > 0 >= sign * (previous.Close - edge.iloc[-2]):
            add("ichimoku", direction, "Close crosses full 9/26/52 cloud; spans shifted forward 26 bars causally", edge.iloc[-1])
        low14, high14 = d.Low.rolling(14).min(), d.High.rolling(14).max()
        stochastic = (100 * (d.Close - low14) / (high14 - low14).replace(0, np.nan)).rolling(3).mean()
        threshold = 20 if sign == 1 else 80
        if sign * (stochastic.iloc[-1] - threshold) > 0 >= sign * (stochastic.iloc[-2] - threshold):
            add("stochastic", direction, f"Slow stochastic 14/3 returns through {threshold}")
        obv = (np.sign(d.Close.diff()).fillna(0) * d.Volume).cumsum()
        obv_edge = obv.shift().rolling(20).max() if sign == 1 else obv.shift().rolling(20).min()
        if sign * (obv.iloc[-1] - obv_edge.iloc[-1]) > 0 >= sign * (obv.iloc[-2] - obv_edge.iloc[-2]):
            add("obv", direction, "OBV crosses prior 20-bar extreme; directional-volume proxy, not investor identity")
    if now.High <= previous.High and now.Low >= previous.Low and now.High - now.Low < previous.High - previous.Low:
        add("inside", "neutral", "Range contained within previous candle; awaiting breakout", status="watch")
    if now.High > now.Low and abs(now.Close - now.Open) <= .1 * (now.High - now.Low):
        add("doji", "neutral", "Body <=10% of range; no standalone directional interpretation", status="context")
    baseline = d.atr.iloc[-21:-1].mean()
    if baseline > 0 and atr >= 1.25 * baseline:
        add("atr", "neutral", f"ATR14 / preceding 20 ATR values = {atr / baseline:.3f}", status="context")
    sar, directions = _sar(d)
    if directions[-1] != directions[-2]:
        add("sar", "bullish" if directions[-1] else "bearish", "SAR flip: acceleration 0.02 to 0.20; prior-two-bar clamp", sar[-1])
    # Fibonacci is context only, anchored to confirmed extrema, not a buy/sell vote.
    highs, lows = pivots(d.High.to_numpy()), pivots(-d.Low.to_numpy())
    if highs and lows:
        hi, lo = highs[-1], lows[-1]
        top, bottom = d.High.iloc[hi], d.Low.iloc[lo]
        if top > bottom:
            for ratio in (.382, .5, .618):
                level = top - ratio * (top - bottom) if hi > lo else bottom + ratio * (top - bottom)
                if abs(price - level) <= .25 * atr:
                    add("fibonacci", "neutral", f"Within 0.25 ATR of {ratio:.1%} retracement between confirmed extrema; 50% is conventional", level, status="context")
    pivot = (previous.High + previous.Low + previous.Close) / 3
    levels = {"P": pivot, "R1": 2 * pivot - previous.Low, "S1": 2 * pivot - previous.High,
              "R2": pivot + previous.High - previous.Low, "S2": pivot - previous.High + previous.Low}
    add("pivot_points", "neutral", {k: float(v) for k, v in levels.items()}, pivot, status="context")
    if now.rvol >= 3 and now.High - now.Low >= 2 * atr:
        add("climax", "neutral", "Relative volume >=3x and range >=2 ATR; exhaustion or continuation unresolved", status="watch")
    if benchmark is not None:
        aligned = pd.concat([d.Close.rename("stock"), benchmark.Close.rename("benchmark")], axis=1, join="inner").dropna()
        if len(aligned) >= 64:
            stock_return = aligned.stock.iloc[-1] / aligned.stock.iloc[-64] - 1
            market_return = aligned.benchmark.iloc[-1] / aligned.benchmark.iloc[-64] - 1
            spread = stock_return - market_return
            if abs(spread) >= .05:
                add("relative_strength", "bullish" if spread > 0 else "bearish",
                    {"stock_63_bar_return": float(stock_return), "benchmark_63_bar_return": float(market_return),
                     "excess_return": float(spread), "note": "Outperformance does not establish investor identity"})
    return events


def evaluate_stock(bars, benchmark=None, asof=None, supplemental=None):
    """One row per specialist, including abstentions and evaluated no-matches."""
    if asof is not None:
        bars = bars.loc[:asof]
        if benchmark is not None:
            benchmark = benchmark.loc[:asof]
    df = validate_bars(bars)
    latest = str(df.index[-1].date()) if len(df) else None
    if len(df) < 60:
        return {"date": latest, "context": {}, "bars": len(df), "agents": [
            {**spec, "status": "insufficient_history", "direction": None, "fit_score": None,
             "evidence": "At least 60 completed bars required", "events": []} for spec in CATALOG]}
    base = scan(df)
    d = indicators(df)
    benchmark_reason = None
    if benchmark is not None:
        benchmark = validate_bars(benchmark)
        if benchmark.empty or benchmark.index[-1] != df.index[-1]:
            benchmark_reason, benchmark = "Benchmark date does not match stock date", None
    extras = _extra_events(d, benchmark)
    specialist = {**daily_rules(d), **supplemental_rules(d, supplemental)}
    rows = []
    for spec in CATALOG:
        rule = spec["rule"]
        row = {**spec, "status": "no_match", "direction": "neutral", "fit_score": 0., "evidence": [], "events": []}
        if spec["id"] in specialist:
            row.update(specialist[spec["id"]])
        elif rule in ABSTENTIONS:
            row.update(status=ABSTENTIONS[rule][0], evidence=ABSTENTIONS[rule][1], direction=None, fit_score=None)
        elif rule == "relative_strength" and (benchmark is None or len(df.index.intersection(benchmark.index)) < 64):
            row.update(status="missing_data", evidence=benchmark_reason or "Requires 64 aligned completed stock/benchmark bars", direction=None, fit_score=None)
        elif rule == "ichimoku" and len(df) < 79:
            row.update(status="insufficient_history", evidence="79 completed bars required for full cloud crossing", direction=None, fit_score=None)
        elif rule == "ma" and len(df) < 201:
            row.update(status="insufficient_history", evidence="201 bars required to evaluate full 50/200 crossover family", direction=None, fit_score=None)
        else:
            if rule in ("wedge", "ascending", "descending", "symmetrical"):
                token = {"wedge": "wedge", "ascending": "Ascending", "descending": "Descending", "symmetrical": "Symmetrical"}[rule]
                events = [s for s in base["signals"] if s["strategy"] == "triangle_wedge" and token in s["variant"]]
            elif rule == "ma":
                events = [s for s in base["signals"] if s["strategy"] in ("sma_cross", "golden_death_cross", "ema_cross")]
            elif rule == "mean_reversion":
                events = [s for s in base["signals"] if s["strategy"] in ("rsi_recovery", "bollinger_reentry")]
            else:
                events = [s for s in base["signals"] if s["strategy"] == rule] + extras.get(rule, [])
            if events:
                directions = set(s["direction"] for s in events)
                direction = next(iter(directions)) if len(directions) == 1 else "mixed"
                statuses = {s.get("status", "matched") for s in events}
                status = "matched" if "matched" in statuses else "watch" if "watch" in statuses else "context"
                # Explicit rule-fit score: match base 60 + participation 20 + proximity 20.
                # WATCH and CONTEXT rows are never directional votes in the judge.
                rvol = base["context"]["rvol"] or 0.
                atr = float(d.atr.iloc[-1])
                distance = min(abs(float(d.Close.iloc[-1]) - s["level"]) for s in events)
                fit = 60 + 20 * min(rvol / 2, 1) + 20 * max(0, 1 - distance / (2 * atr)) if atr > 0 else 60
                if status != "matched":
                    fit = 40. if status == "watch" else 20.
                row.update(status=status, direction=direction, fit_score=round(fit, 2),
                           evidence=[s["evidence"] for s in events], events=events)
        rows.append(row)
    return {"date": latest, "bars": len(df), "context": base["context"], "agents": rows,
            "coverage_notes": base["coverage_notes"],
            "price": float(df.Close.iloc[-1]),
            "dollar_volume_20": float((df.Close * df.Volume).tail(20).mean())}


def judge(panel, bars):
    """Directional evidence triage without legacy trade-qualification vetoes.

    Correlated techniques share one family vote. Bearish evidence stays visible.
    There is no probability score, and missing sources never become bullish.
    """
    df = validate_bars(bars.loc[:panel["date"]])
    bullish, bearish = {}, {}
    for row in panel["agents"]:
        if row["status"] != "matched":
            continue
        for direction, groups in (("bullish", bullish), ("bearish", bearish)):
            if any(s["direction"] == direction for s in row["events"]):
                groups[row["family"]] = max(groups.get(row["family"], 0), row["fit_score"])
    evaluated = sum(row["status"] in ("matched", "no_match", "watch", "context") for row in panel["agents"])
    result = {"decision": "low_priority", "evaluated_agents": evaluated, "abstaining_agents": 50 - evaluated,
              "bullish_families": bullish, "bearish_families": bearish,
              "supporting_family_count": len(bullish), "opposing_family_count": len(bearish),
              "reasons": [], "entry": None, "invalidation": None, "nearest_resistance": None,
              "reward_risk": None, "net_reward_risk_10bps": None, "playbook": "unclassified",
              "research_checks": ["Verify earnings/news catalysts and gap risk", "Review sourced bull/bear chatter and freshness",
                                  "Refresh prices and confirm levels", "Inspect pattern chart; heuristic detection can fail"],
              "chatter_status": "not_checked"}
    result["policy"] = "evidence_only_v2"
    result["risk_context_only"] = True
    result["evidence_direction"] = ("mixed" if bullish and bearish else "bullish" if bullish
                                    else "bearish" if bearish else "neutral")
    if not evaluated:
        result.update(decision="insufficient_data", reasons=["No evaluable specialist data"])
        return result
    if bullish or bearish:
        result["decision"] = "research_priority"
        result["reasons"] = ["Matched directional evidence; no legacy trade filters applied"]
    elif any(r["status"] == "watch" for r in panel["agents"]):
        result["decision"] = "watch"
        result["reasons"] = ["Developing pattern; no confirmed directional match"]
    else:
        result["reasons"] = ["No fresh directional match"]
    regime = panel["context"].get("regime")
    result["playbook"] = ("mixed_evidence" if bullish and bearish else "bearish_research" if bearish
                          else "trend_continuation" if bullish and regime == "uptrend"
                          else "recovery" if bullish else "unclassified")
    atr = panel["context"].get("atr")
    if atr is None or not np.isfinite(atr) or atr <= 0 or df.empty:
        return result
    # Legacy long-side scenario retained only as descriptive context, never ranking input.
    entry = float(df.High.iloc[-1] + .05 * atr)
    stop = float(df.Low.tail(3).min() - .25 * atr)
    risk = entry - stop
    h = df.High.to_numpy()
    obstacles = [float(h[i]) for i in range(max(1, len(df) - 252), len(df) - 1)
                 if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > entry]
    resistance = min(obstacles) if obstacles else None
    rr = (resistance - entry) / risk if resistance is not None and risk > 0 else None
    net_rr = ((resistance - entry) - entry * .001) / risk if rr is not None else None
    result.update(entry=entry, invalidation=stop, nearest_resistance=resistance,
                  reward_risk=rr, net_reward_risk_10bps=net_rr)
    return result


def rank_agent_rows(rows):
    """Within-agent ranks only among matched rows with the SAME direction.

    No-match/abstention/context/watch entries have no directional match rank.
    Equal scores get the same rank. Nothing is labelled a win probability.
    """
    groups = defaultdict(list)
    for row in rows:
        row["agent_rank"] = None
        if row["status"] == "matched" and row["direction"] in ("bullish", "bearish"):
            groups[(row["id"], row["direction"])].append(row)
    for members in groups.values():
        scores = sorted((r["fit_score"] for r in members), reverse=True)
        for row in members:
            row["agent_rank"] = scores.index(row["fit_score"]) + 1
    return rows


def research_sort_key(row):
    """Rank available directional evidence; risk metrics and conflicts never veto."""
    tier = {"research_priority": 0, "watch": 1, "low_priority": 2, "insufficient_data": 3}
    bull, bear = row.get("bullish_families", {}), row.get("bearish_families", {})
    families = {k: max(bull.get(k, 0), bear.get(k, 0)) for k in set(bull) | set(bear)}
    fit = sum(families.values()) / len(families) if families else 0
    return (tier[row["decision"]], -max(len(bull), len(bear)), -len(families),
            -fit, row["market_cap_rank"])
