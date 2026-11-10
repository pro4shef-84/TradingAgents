# Specialist rules v1 (2026-11-09)

Deterministic research definitions, not validated alpha. Original fifty IDs and
family caps are retained. New rules carry `specialist_rules_v1`, observed window,
confirmation time, levels, and parameters in their evidence. Unconfirmed shapes
are watch observations. All windows end at the requested cutoff.

## Daily causal proxies

Prices below are close prices, ATR is trailing ATR14, and all comparisons use
only the last completed window. Pivots are extrema confirmed by three bars on
each side (never the unconfirmed final three bars).

* 5 rounding bottom: preceding 40 closes, split into four ten-bar means;
  outer means within 2 ATR, middle means at least 1 ATR below both outer
  means, first-half slope negative and second-half slope positive. Current
  close above prior-window maximum confirms; otherwise watch.
* 6 diamond: preceding 32 high/low ranges split into four eight-bar widths;
  second width >=1.5 first and fourth <=0.75 second, third <=second. Latest
  close outside the prior eight-bar range confirms in that direction;
  otherwise watch. This is a range-expansion/contraction proxy.
* 12 cup/handle: 40-bar cup followed by eight-bar handle, excluding current
  bar. Cup first/last five-close means within 1 ATR, middle ten-close mean
  >=2 ATR below rim; cup halves descend/ascend; handle depth <=half cup
  depth and >=0.25 ATR, handle stays above cup midpoint. Close above rim
  confirms, otherwise watch.
* 14 measuring gap: current full candle gaps >=0.25 ATR beyond the prior
  candle following a ten-bar move >=3 ATR in the same direction; volume
  >=1.5 preceding 20-bar mean. Gap boundary is trigger/invalidation reference.
* 25 **Price-zone reaction proxy** (historical name: Order blocks): last
  confirmed three-sided low/high pivot in preceding 20 bars, zone +/-0.25 ATR;
  latest low/high touches zone and close rejects it by >=0.5 ATR in the
  direction away from the pivot. No ownership/order-intent inference.
* 39 Wyckoff range proxy: prior 20-bar range <=8 ATR; latest price probes
  its boundary by >=0.1 ATR and closes back inside by >=0.25 ATR on
  >=1.5 prior-20 volume. Spring/upthrust proxy, not proof of accumulation.

## Normalized supplemental input

`evaluate_stock(bars, benchmark=None, asof=None, supplemental=None)` remains
compatible. Supplemental is a dict with aware ISO `cutoff`, `symbol`, and dataset
keys `intraday`, `trades`, `quotes`, `book`, `events`, `pairs`. Each dataset is a
mapping: `provider`, `feed`, `symbol`, `session` (YYYY-MM-DD), `retrieved_at`
(aware ISO), `adjustment`, `coverage` (mapping with `complete` bool), `records`
(list). Missing data explicitly abstains. `cutoff` is independent of daily asof.
Records require aware `timestamp` (alias `event_at`) and `available_at`; both
must be <=cutoff. Optional `updated_at` raises availability to the revision time.
Future records are ignored before validation. Retrieval time can be later than
cutoff (historical reconstruction); availability cannot. Duplicate timestamps
are invalid for bars, while simultaneous trades/quotes are valid. A top-level
symbol, when provided, must agree with dataset symbols. No naive timestamps.

Intraday records use bar-start `timestamp`, `interval_minutes`, lower-case
`open/high/low/close/volume`, optional genuine provider `vwap`. Bars must finish
before cutoff and availability; OHLC and positive volume are validated. Regular
session boundaries come from the XNYS exchange calendar, including holidays, DST,
and early closes. Optional `session_close` must agree with that calendar. Session
date must agree. Dataset
coverage must be complete for intraday evaluation; all intervals from open to
latest bar must be present. Intraday events explicitly carry `timeframe=intraday`.

* 15 ORB: first 30 minutes, then latest completed bar closes beyond opening
  high/low; previous post-opening close must not already be beyond that side.
* 30 VWAP: latest close crosses volume-weighted provider bar VWAP; if any
  provider VWAP is missing use HLC3 for all bars and label approximate.
* 44 Market Profile: at least two complete regular sessions, 30-minute
  blocks; each block contributes one observation at its HLC3 (time-at-price,
  not volume). Fixed price bins of `bin_size` (default 0.25 dollars), sorted
  count descending with ascending-price tie break, take bins until >=70%
  observations; both latest session VA bounds above/below prior confirms.
* 46 gap/go: regular open >=1% from the latest daily close strictly before
  the intraday session date and close
  at/after first 30 minutes remains >=0.5% beyond open in gap direction.
  Explicit `premarket_observed` defaults false. Declared daily/intraday
  adjustment mismatches and unknown daily adjustment abstain. Daily adjustment
  can be set in DataFrame attrs. Alternatively `supplemental.previous_close` is
  `{price,asof,adjustment,provider,feed}`, using the immediately previous XNYS
  date as `asof` and the same adjustment as intraday bars. A sourced raw previous
  close allows comparison when the daily series uses adjusted prices.
* 38 volume profile: valid positive trade price/size, full declared coverage;
  fixed floor(price/bin_size) bins, volume sum, lowest-price tie break POC;
  latest trade >=one bin above/below POC. Partial coverage is abstention.
* 42 tape: bounded observed coverage is allowed and retained in evidence for
  trades and quotes; this never certifies a complete tape. >=10 eligible trades, classified at most recent preceding quote
  <=2 seconds old, never a future quote. Bid/ask midpoint side, midpoint
  trades unclassified. >=80% classified, >=65% buy/sell volume confirms.
  Conditions are conservatively limited to empty lists or exactly `["@"]`
  for provider `Alpaca` (regular sale); unknown/combined conditions and
  `correction` or `cancelled` prints are excluded;
  correction/cancellation `trade_id` (or `original_trade_id`) also removes the
  associated original print;
  quote bid<=ask and both positive. Coverage limitations are explicit.
* 43 book: full snapshot initializes sequence, every subsequent update must
  increment sequence by one; any gap invalidates until fresh snapshot.
  Snapshot imports need not claim complete session coverage; coverage stays visible.
  Snapshots/updates contain full current `bids`/`asks` [price,size] arrays;
  top five summed depth imbalance >=0.6 or <=-0.6 confirms. No inferred live
  entitlement. Empty/crossed/invalid books abstain.
* 47 catalyst: event `verified` must be literal true with `source_url`,
  `event_time`, aware `available_at`, direction bullish/bearish. Reviewed imports
  may have partial source coverage, retained explicitly. At least
  two completed intraday bars starting after availability, and >=1% move
  from first postavailability open in event direction. Unverified keywords
  never qualify; absent reviewed events yields missing_evidence.
* 50 pairs: explicit dataset `pair` [stock,partner], `adjustment=adjusted`,
  `cost_bps` >=0 and `borrow_assumption` string. Records have aligned
  `stock`,`partner` adjusted positive prices. Last 60 observations form the
  model, excluding latest five trading observations. OLS log-stock on
  log-partner, beta 0.2–5, nonzero residual standard deviation; residual
  lag-one correlation <0.95 in each 30-row formation half and half beta
  difference <=50% full beta. Latest trading residual z >=2 or <=-2 yields
  contextual pair-spread observation, never an outright stock vote. Frozen
  formation model, no trading-period refit. Unknown borrow remains research
  only and prevents executable-return claims.

All supplemental matches expose coverage/provider/feed and availability cutoff.
No rule interprets a missing input as neutral. JSON output contains no NaN/Inf.

## Historical specialist validation

`scripts/validate_specialists.py` runs a frozen JSON protocol against saved daily
CSV files. The protocol specifies `symbols`, `start`, `formation_end`, `end`,
`holding_bars`, `cost_bps`, input hashes (including SPY) and evaluator code hashes.
The `shortlist` and `control` arrays label the two comparison cohorts.

```sh
.venv/bin/python scripts/validate_specialists.py \
  --prices results/daily_research/2026-11-09/prices \
  --protocol results/specialist_validation_2026-11-09/protocol.json \
  --output results/specialist_validation_2026-11-09
```

The runner refuses to overwrite an existing validation result or accept changed
frozen inputs. It writes full observations, rule/direction summaries and an HTML
report. Signals use completed prefixes only. Entry is next-bar open, exit is the
close of the specified holding bar. Round-trip costs are subtracted from both
the stock return and long SPY over the identical entry/exit dates. Missing SPY
dates leave excess returns unavailable. Bearish observations exclude borrow
costs and locate constraints, and cannot establish executable short returns.
Pairs rule 50 is excluded from outright stock returns.

Formation windows must exit on or before the formation cutoff; evaluation
signals must occur after it. Boundary-crossing windows are excluded. Rule and
direction groups are selected using formation data only: at least 20 paired
observations, positive mean net return, and positive mean excess over SPY.
Every evaluation group remains visible, with formation selection marked.

These are overlapping signal-window observations, not an invested portfolio or
independent trials. The historical sample uses surviving symbols, adjusted
prices as retrieved today, and no archived supplemental evidence. A chronological
split is not certified previously unseen data. Today's shortlist is selected
retrospectively; its historical performance cannot validate the current shortlist
selection policy. Prospective archived reports must supply that evidence.

Alpaca regular-sale condition reference: [official market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq), “How are bars aggregated?”
Only the explicit regular-sale condition above is allowed; this does not certify
that historical data contains every correction or every market venue.
