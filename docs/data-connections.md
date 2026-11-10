# Supplemental research data connections

All acquisition is read-only. No subscription purchases, orders, personal Discord tokens, social posting, or credential values appear in the output.

## Alpaca historical market data

Official API references checked November 9, 2026:

- [Historical bars](https://docs.alpaca.markets/us/reference/stockbars): `GET https://data.alpaca.markets/v2/stocks/bars`, `timeframe=1Min`, `adjustment=raw`.
- [Historical trades](https://docs.alpaca.markets/us/reference/stocktrades-1): `GET /v2/stocks/trades`.
- [Historical quotes](https://docs.alpaca.markets/us/reference/stockquotes-1): `GET /v2/stocks/quotes`.
- [Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq): feed access depends on subscription and recency. Historical SIP access does not prove current real-time entitlement.

All three endpoints accept explicit symbols/start/end/feed/limit/sort, then `page_token` from `next_page_token`. Results sort by symbol first, so a page limit alone does not guarantee coverage for every symbol. Our adapter follows tokens up to a hard budget, rejects repeated tokens, and reports truncation for every symbol when that budget expires. HTTP 401/403 raises `AlpacaEntitlementError` (status `insufficient_entitlement`); caller must distinguish authentication problems from account feed licensing. There is no silent IEX substitution. Network errors, 429 and 5xx retry at most three attempts with bounded delays. Errors never include response bodies, request headers or credentials.

Credentials continue to use environment/local `.env` `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. Configuration alone reports `configured`; only a timestamped successful observation can report `connected`, and it expires to `stale`. Live depth, X and Stocktwits remain `not_connected`.

```python
from tradingagents.pilot.market_inputs import fetch_intraday, fetch_trades, fetch_quotes, for_symbol
batch = fetch_intraday(['SPY'], '2026-11-06T14:30:00Z', '2026-11-06T15:01:00Z', feed='sip')
per_symbol = for_symbol(batch, 'SPY')
# evaluate_stock(..., supplemental={'symbol':'SPY', 'cutoff': ..., 'intraday':per_symbol})
```

Each fetch accepts `max_pages=10`, `page_limit=1000`; maximum 20 input symbols, seven-day window, 100 pages, 10,000 records/page. Requests are one multi-symbol batch; the daily pipeline separately restricts enrichment to 20 candidates. Returned `budget` declares the selected caps. Callers should select short tick windows because liquid symbols exhaust the record budget quickly.

The batch envelope contains provider/feed/adjustment/retrieved_at/start/end, `records` and `coverage` maps keyed by symbol. `for_symbol` projects it into the evaluator envelope. Records contain provider, feed, symbol, UTC `event_at`/`timestamp`, UTC `available_at`, retrieval time, raw adjustment policy and New York session date. Bars have lower-case open/high/low/close/volume/vwap and `interval_minutes=1`; trades retain price/size/conditions/trade_id/exchange/correction; quotes contain bid/ask/bid_size/ask_size/conditions. Exact duplicates are removed; conflicting duplicate bars raise an error. Malformed times, naive times, impossible OHLC and crossed quotes raise errors.

Bar coverage enumerates every missing clock minute inside the requested interval, including extended hours. Missing zero-trade minutes are not fabricated. Request regular-session windows from the exchange calendar for regular-session analysis; the adapter does not infer market holidays. Incomplete final minute bars are excluded. Complete bar coverage means only every requested clock minute was returned. Tick coverage is always `observed` or `partial`, never certified complete. Tick conditions/corrections are preserved for downstream filtering; historical APIs cannot certify a correction-free original tape. Historical availability is a conservative bar-end/tick-time proxy; original publication/revision times are unavailable and retrospective historical downloads are not an archive of originally observed ticks. Bar VWAP is provider-supplied; OHLC does not manufacture exact VWAP.

## Verified bounded capability check

Existing credentials were used November 9, 2026, 16:55:38–39 UTC. One request/page per endpoint, 100 record limit, one SPY minute from November 6 14:30–14:31 UTC:

| Capability | Observed | Scope |
| --- | --- | --- |
| SIP minute bars | connected, 1 bar, no missing requested minutes | historical minute only |
| SIP trades | connected, 100 records, truncated | partial historical tape |
| SIP quotes | connected, 100 records, truncated | partial historical quotes |

No live full-universe refresh, current real-time SIP entitlement, market-depth access, or trade execution was tested. A later check may return different entitlement or freshness status.

## Sourced imports: reviewed events, depth and optional social

`import_records(source, kind='events'|'books'|'social', cutoff=...)` accepts a list or JSON/CSV path. Limits: 10 MB file and 10,000 rows. Every row requires symbol, timezone-aware event_at (or event_time/timestamp), available_at, provider and HTTPS source_url. `updated_at`, when present, raises effective availability to the edit time; later edits never appear before their availability cutoff. Imports retain source links, session, feed, retrieval and coverage. Imports report `imported`, `complete=false`; validation does not independently prove source completeness or truth. Event records require explicit boolean `verified` (CSV lower-case `true`/`false` accepted). A headline classifier is never verification. Direction and descriptive event details may be supplied by the reviewed source.

```json
[{"symbol":"SPY","provider":"reviewed official release","source_url":"https://example.com/release","event_at":"2026-11-06T14:30:00Z","available_at":"2026-11-06T14:31:00Z","verified":true,"direction":"bullish"}]
```

Books require integer `sequence`, `type: snapshot|update` (or boolean snapshot), full resulting `bids`/`asks` arrays of `[price,size]`, positive prices and uncrossed sides. Updates are full state, not deltas. The first state must be a snapshot. A sequence gap invalidates subsequent updates until an explicit snapshot. Book timestamps must be ordered per symbol. CSV arrays must be JSON strings. Actual licensed live depth requires a configured supported provider and independent connection testing; no live depth provider is claimed here.

X and Stocktwits live collection is disabled/unconnected; sourced social imports provide an honest supported route without invented endpoints or access claims. A future live adapter requires official documented account access, explicit bounded request budgets and a verified connection. Imported social text remains observations, not sentiment or trade evidence.

## Discord bot reader

The existing authorized bot-only REST reader now uses bounded `before` pagination, at most three pages (300 messages) per channel by default and hard limit ten pages; at most five configured channels. It rejects repeated cursors and malformed IDs, removes duplicate message IDs, excludes future edited content and reports messages/pages/truncation and possibly missing content. Historical completeness remains unverified even if pagination ends. [Official Message API documentation](https://docs.discord.com/developers/resources/message#get-channel-messages) confirms channel-message pagination and the privileged Message Content intent; missing intent can produce empty content. No private Discord connection was verified in the Alpaca smoke check. Access requires the authorized bot token, channel IDs, channel history permissions and appropriate content intent. No personal-token route or outgoing message capability is provided.

The existing daily bar adapter also caps pagination at 100 pages and rejects budget exhaustion instead of saving partial daily inputs.
