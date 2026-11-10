"""Deterministic, completed-bar technical screens; no orders or profitability claims.

All lookbacks are in bars. Input OHLC must share the same adjustment convention.
Chart patterns are explicit heuristics, not universal definitions.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

CATALOG = {
    "sma_cross": "Price crosses the 20, 50 or 200 SMA",
    "sma_bounce": "20/50/200 SMA touch and rejection in the average's trend direction",
    "golden_death_cross": "50 SMA crosses 200 SMA",
    "ema_cross": "9 EMA crosses 21 EMA",
    "donchian_breakout": "Close breaks preceding 20 or 55 bar extreme",
    "breakout_retest": "Prior 20-bar breakout retested and held",
    "macd_cross": "12/26 MACD crosses its 9 EMA signal line",
    "rsi_recovery": "Wilder RSI(14) returns from below 30 or above 70",
    "bollinger_reentry": "Close returns inside 20-bar, 2 standard deviation bands",
    "squeeze_breakout": "Bollinger bandwidth contraction followed by band breakout",
    "engulfing": "Opposite-colour candle body engulfed after a preceding trend",
    "hammer_star": "Long-wick hammer or shooting star after a preceding trend",
    "volume_breakout": "20-bar breakout with at least 1.5x prior 20-bar mean volume",
    "double_top_bottom": "Confirmed double top/bottom with fresh neckline break",
    "head_shoulders": "Confirmed head and shoulders/inverse with fresh neckline break",
    "triangle_wedge": "Converging confirmed pivot boundaries with fresh breakout",
}


@dataclass(frozen=True)
class Signal:
    strategy: str
    variant: str
    direction: str
    date: str
    close: float
    level: float
    invalidation: float
    evidence: dict

    def to_dict(self):
        return asdict(self)


def validate_bars(bars):
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("Use a DatetimeIndex of completed bars")
    if bars.index.hasnans or bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique, valid and increasing")
    if not set(required).issubset(bars.columns):
        raise ValueError(f"Required columns: {required}")
    df = bars[required].astype(float).copy()
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("OHLCV must be finite; missing bars must be resolved upstream")
    if (df[required[:4]] <= 0).any().any() or (df.Volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    # CSV/adjustment round trips can move an equal high/close by a few ULPs.
    # Permit only machine-precision differences, never a price-sized correction.
    epsilon = 16 * np.finfo(float).eps * df[required[:4]].abs().max(axis=1)
    if (((df[["Open", "Close", "Low"]].max(axis=1) - df.High) > epsilon) |
            ((df.Low - df[["Open", "Close", "High"]].min(axis=1)) > epsilon)).any():
        raise ValueError("Inconsistent OHLC bounds")
    return df


def wilder(series, period=14):
    """Wilder smoothing seeded with the first period's arithmetic mean."""
    out = pd.Series(np.nan, index=series.index)
    valid = series.dropna()
    if len(valid) >= period:
        seed = valid.iloc[:period].mean()
        tail = valid.iloc[period - 1:].copy()
        tail.iloc[0] = seed
        out.loc[tail.index] = tail.ewm(alpha=1 / period, adjust=False).mean()
    return out


def indicators(df):
    c, h, low = df.Close, df.High, df.Low
    out = df.copy()
    for p in (20, 50, 200):
        out[f"sma{p}"] = c.rolling(p).mean()
    for p in (9, 12, 21, 26):
        out[f"ema{p}"] = c.ewm(span=p, adjust=False, min_periods=p).mean()
    out["macd"] = out.ema12 - out.ema26
    out["macd_signal"] = out.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    delta = c.diff()
    gain, loss = wilder(delta.clip(lower=0)), wilder(-delta.clip(upper=0))
    out["rsi"] = 100 - 100 / (1 + gain / loss)
    out.loc[(gain == 0) & (loss == 0), "rsi"] = 50
    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = wilder(tr)
    up, down = h.diff(), -low.diff()
    plus = wilder(up.where((up > down) & (up > 0), 0))
    minus = wilder(down.where((down > up) & (down > 0), 0))
    dx = (100 * (plus - minus).abs() / (plus + minus)).where(plus + minus != 0, 0)
    out["adx"] = wilder(dx)
    std = c.rolling(20).std(ddof=0)
    out["upper"], out["lower"] = out.sma20 + 2 * std, out.sma20 - 2 * std
    out["bandwidth"] = (out.upper - out.lower) / out.sma20
    out["rvol"] = df.Volume / df.Volume.shift().rolling(20).mean().replace(0, np.nan)
    return out


def pivots(values, radius=3):
    """A pivot is available only after radius subsequent bars have completed."""
    return [i for i in range(radius, len(values) - radius)
            if values[i] > max(values[i - radius:i])
            and values[i] > max(values[i + 1:i + radius + 1])]


def scan(bars, asof=None):
    """Return latest-bar events and context, with optional point-in-time cutoff."""
    if asof is not None:
        bars = bars.loc[:asof]
    df = validate_bars(bars)
    if len(df) < 60:
        raise ValueError("At least 60 completed bars required; 201+ for 200-SMA events")
    d = indicators(df)
    now, prev = d.iloc[-1], d.iloc[-2]
    a, price = float(now.atr), float(now.Close)
    signals = []

    def add(key, variant, sign, level, stop, **evidence):
        if not all(np.isfinite(v) for v in (level, stop)):
            return
        signals.append(Signal(key, variant, "bullish" if sign == 1 else "bearish",
                              str(d.index[-1].date()), price, float(level), float(stop), evidence))

    def cross(left, right, sign):
        return sign * (left.iloc[-1] - right.iloc[-1]) > 0 and sign * (left.iloc[-2] - right.iloc[-2]) <= 0

    for sign in (1, -1):
        stop = price - sign * 2 * a
        for p in (20, 50, 200):
            ma = d[f"sma{p}"]
            if cross(d.Close, ma, sign):
                add("sma_cross", f"SMA {p}", sign, ma.iloc[-1], stop, period=p)
            extreme = now.Low if sign == 1 else now.High
            if (a > 0 and abs(extreme - ma.iloc[-1]) <= .3 * a
                    and 0 < sign * (price - ma.iloc[-1]) <= a
                    and sign * (ma.iloc[-1] - ma.iloc[-6]) > 0
                    and sign * (prev.Close - ma.iloc[-2]) > 0
                    and sign * (price - now.Open) > 0):
                add("sma_bounce", f"SMA {p}", sign, ma.iloc[-1], extreme - sign * .25 * a,
                    period=p, tolerance_atr=.3)
        for key, variant, left, right in (
            ("golden_death_cross", "Golden cross" if sign == 1 else "Death cross", d.sma50, d.sma200),
            ("ema_cross", "EMA 9/21", d.ema9, d.ema21),
            ("macd_cross", "MACD 12/26/9", d.macd, d.macd_signal),
        ):
            if cross(left, right, sign):
                add(key, variant, sign, price, stop,
                    indicator=float(left.iloc[-1]), comparison=float(right.iloc[-1]))
        for period in (20, 55):
            boundary = d.High.shift().rolling(period).max() if sign == 1 else d.Low.shift().rolling(period).min()
            if cross(d.Close, boundary, sign):
                add("donchian_breakout", f"{period}-bar breakout", sign, boundary.iloc[-1], stop, period=period)
                if period == 20 and now.rvol >= 1.5:
                    add("volume_breakout", "20-bar breakout", sign, boundary.iloc[-1], stop, relative_volume=float(now.rvol))
        threshold = 30 if sign == 1 else 70
        if sign * (now.rsi - threshold) > 0 >= sign * (prev.rsi - threshold):
            add("rsi_recovery", "RSI 14", sign, price, stop, rsi=float(now.rsi))
        band = d.lower if sign == 1 else d.upper
        if cross(d.Close, band, sign):
            add("bollinger_reentry", "Bollinger 20/2", sign, band.iloc[-1], stop)
        outer = d.upper if sign == 1 else d.lower
        contraction = d.bandwidth.shift().rolling(40).quantile(.2).iloc[-1]
        if prev.bandwidth <= contraction and cross(d.Close, outer, sign):
            add("squeeze_breakout", "Bollinger squeeze", sign, outer.iloc[-1], stop,
                prior_bandwidth=float(prev.bandwidth))
        prior_trend = sign * (d.Close.iloc[-2] - d.Close.iloc[-7]) < -a
        if (prior_trend and sign * (prev.Close - prev.Open) < 0
                and sign * (price - now.Open) > 0
                and sign * (price - prev.Open) >= 0
                and sign * (now.Open - prev.Close) <= 0):
            add("engulfing", "Bullish engulfing" if sign == 1 else "Bearish engulfing",
                sign, price, (now.Low if sign == 1 else now.High) - sign * .25 * a)
        body = abs(price - now.Open)
        lower_wick = min(price, now.Open) - now.Low
        upper_wick = now.High - max(price, now.Open)
        long_wick, short_wick = (lower_wick, upper_wick) if sign == 1 else (upper_wick, lower_wick)
        if prior_trend and body > 0 and long_wick >= 2 * body and short_wick <= body and now.High - now.Low >= a:
            add("hammer_star", "Hammer" if sign == 1 else "Shooting star", sign,
                price, (now.Low if sign == 1 else now.High) - sign * .25 * a)
        # Require a fresh retest on this bar, following a breakout 3–9 bars ago.
        x = sign * d.Close.to_numpy()
        upper = d.High.to_numpy() if sign == 1 else -d.Low.to_numpy()
        lower = d.Low.to_numpy() if sign == 1 else -d.High.to_numpy()
        for b in range(len(d) - 10, len(d) - 2):
            level = max(upper[b - 20:b])
            if (x[b] > level >= x[b - 1] and min(lower[b + 1:-1]) > level + .3 * a
                    and abs(lower[-1] - level) <= .3 * a and x[-1] > level):
                add("breakout_retest", "20-bar retest", sign, sign * level,
                    sign * (lower[-1] - .25 * a), breakout_date=str(d.index[b].date()))
                break
    if a > 0:
        _patterns(d, a, add)
    context = {k: (float(now[k]) if np.isfinite(now[k]) else None)
               for k in ("sma20", "sma50", "sma200", "rsi", "atr", "adx", "rvol")}
    context["regime"] = ("unknown" if context["sma200"] is None else
                         "uptrend" if price > now.sma50 > now.sma200 else
                         "downtrend" if price < now.sma50 < now.sma200 else "mixed")
    return {"date": str(d.index[-1].date()), "bars": len(d), "context": context,
            "coverage_notes": [] if len(d) >= 205 else ["Some 200-SMA rules lack warmup history"],
            "signals": [s.to_dict() for s in signals]}


def _patterns(d, atr, add):
    # Local close pivots; only the most recent structure is tested.
    c = d.Close.to_numpy()[-150:]
    n = len(c)
    for flip in (1, -1):
        x = flip * c
        peaks = pivots(x)
        sign = -flip

        def emit(key, variant, p, q, level, previous_level, stop, points):
            if x[-1] < level and x[-2] >= previous_level and q <= n - 5:
                add(key, variant, sign, flip * level, flip * stop,
                    pivot_dates=[str(d.index[-n + i].date()) for i in points],
                    pivot_confirmation_bars=3, heuristic=True)

        if len(peaks) >= 2:
            p, q = peaks[-2:]
            valley = p + int(np.argmin(x[p:q + 1]))
            level = x[valley]
            if (10 <= q - p <= 60 and abs(x[p] - x[q]) <= atr
                    and min(x[p], x[q]) - level >= 2 * atr
                    and p >= 10 and x[p] - min(x[max(0, p - 20):p]) >= 2 * atr
                    and min(x[q + 1:-1]) >= level):
                emit("double_top_bottom", "Double top" if flip == 1 else "Double bottom",
                     p, q, level, level, max(x[p], x[q]) + .25 * atr, [p, valley, q])
        if len(peaks) >= 3:
            p, q, r = peaks[-3:]
            v1 = p + int(np.argmin(x[p:q + 1]))
            v2 = q + int(np.argmin(x[q:r + 1]))
            if (20 <= r - p <= 90 and min(q - p, r - q) >= 6
                    and .4 <= (q - p) / (r - q) <= 2.5
                    and abs(x[p] - x[r]) <= atr and x[q] - max(x[p], x[r]) >= atr
                    and min(x[p] - x[v1], x[r] - x[v2]) >= atr
                    and p >= 10 and x[p] - min(x[max(0, p - 20):p]) >= 2 * atr):
                slope = (x[v2] - x[v1]) / (v2 - v1)
                neck = x[v1] + slope * (np.arange(n) - v1)
                if abs(slope) <= .08 * atr and np.all(x[r + 1:-1] >= neck[r + 1:-1]):
                    emit("head_shoulders", "Head and shoulders" if flip == 1 else "Inverse head and shoulders",
                         p, r, neck[-1], neck[-2], x[r] + .25 * atr, [p, v1, q, v2, r])
    # Fit boundaries to at least three confirmed highs/lows of the last 60 bars.
    c = c[-60:]
    highs, lows = pivots(c), pivots(-c)
    if len(highs) < 3 or len(lows) < 3:
        return
    hi, lo = np.array(highs[-3:]), np.array(lows[-3:])
    if min(hi[-1] - hi[0], lo[-1] - lo[0]) < 10:
        return
    hs, hb = np.polyfit(hi, c[hi], 1)
    ls, lb = np.polyfit(lo, c[lo], 1)
    end, start = len(c) - 1, max(hi[0], lo[0])
    upper, lower = hs * end + hb, ls * end + lb
    old_width = (hs - ls) * start + hb - lb
    width = upper - lower
    if not (0 < width < .8 * old_width and hs < ls):
        return
    if max(np.max(abs(c[hi] - (hs * hi + hb))), np.max(abs(c[lo] - (ls * lo + lb)))) > .5 * atr:
        return
    flat = .02 * atr
    if abs(hs) <= flat and ls > flat:
        variant = "Ascending triangle"
    elif abs(ls) <= flat and hs < -flat:
        variant = "Descending triangle"
    elif hs < -flat and ls > flat:
        variant = "Symmetrical triangle"
    elif hs > flat and ls > flat:
        variant = "Rising wedge"
    elif hs < -flat and ls < -flat:
        variant = "Falling wedge"
    else:
        return
    for sign, slope, intercept, boundary in ((1, hs, hb, upper), (-1, ls, lb, lower)):
        if sign * (c[-1] - boundary) > 0 >= sign * (c[-2] - (slope * (end - 1) + intercept)):
            add("triangle_wedge", variant, sign, boundary, lower if sign == 1 else upper,
                heuristic=True, upper_slope=float(hs), lower_slope=float(ls))


def evaluate_events(bars, holding_bars=10, cost_bps=10, start=None):
    """Forward event study, NOT a portfolio backtest. Overlapping events retained.

Signal at completed close t, entry at open t+1, exit close t+holding_bars.
cost_bps is total round-trip commission/slippage per event. No short borrow costs.
"""
    if holding_bars < 1 or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Holding bars must be positive and cost_bps finite/nonnegative")
    df = validate_bars(bars)
    events = []
    for i in range(59, len(df) - holding_bars):
        if start is not None and df.index[i] < pd.Timestamp(start):
            continue
        result = scan(df.iloc[:i + 1])
        entry, exit_price = float(df.Open.iloc[i + 1]), float(df.Close.iloc[i + holding_bars])
        for signal in result["signals"]:
            direction = 1 if signal["direction"] == "bullish" else -1
            events.append({**signal, "entry_date": str(df.index[i + 1].date()),
                           "exit_date": str(df.index[i + holding_bars].date()),
                           "entry": entry, "exit": exit_price,
                           "net_return": direction * (exit_price / entry - 1) - cost_bps / 10000})
    groups = {}
    for event in events:
        key = f"{event['strategy']} / {event['variant']} / {event['direction']}"
        groups.setdefault(key, []).append(event["net_return"])
    summary = {k: {"events": len(v), "mean_net_return": float(np.mean(v)),
                   "median_net_return": float(np.median(v)),
                   "positive_fraction": float(np.mean(np.array(v) > 0))} for k, v in groups.items()}
    return {"method": "overlapping fixed-horizon event study; not portfolio returns",
            "holding_bars": holding_bars, "round_trip_cost_bps": cost_bps,
            "limitations": ["No stop/target execution", "No short borrow or options pricing",
                            "No survivorship correction or statistical significance claim",
                            "Use a held-out date range; do not tune on evaluation results"],
            "summary": summary, "events": events}
