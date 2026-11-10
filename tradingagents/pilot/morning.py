"""Once-a-day, bounded news collection for the frozen top-1,000 universe."""

import pandas as pd
from tradingagents.pilot.chatter import news


def top_tickers(source):
    frame = pd.read_csv(source / 'universe.csv').sort_values('rank')
    tickers = frame.head(1000).ticker.tolist()
    if len(tickers) != 1000 or len(set(tickers)) != 1000:
        raise ValueError('Expected 1,000 unique market-cap ranked symbols')
    return tickers


def collect(tickers, now, since):
    items = {}; errors = []; capped = []; successful = 0
    for offset in range(0, len(tickers), 50):
        batch = tickers[offset:offset+50]
        try:
            rows, status = news(batch, now, since)
            successful += len(batch)
            if status['truncated']: capped.append(batch)
            for row in rows:
                if row['id'] in items:
                    row['symbols'] = sorted(set(row['symbols']) | set(items[row['id']]['symbols']))
                items[row['id']] = row
        except Exception as exc:
            errors.append(dict(tickers=batch, error=str(exc) if isinstance(exc, ValueError) else type(exc).__name__))
    source = dict(status='connected' if not errors and not capped else 'partial' if successful else 'error',
        requested_symbols=len(tickers), successful_symbols=successful, errors=errors,
        truncated_batches=capped, fetched_at=now, window_start=since,
        coverage='50-symbol batches, maximum 500 articles per batch; caps and failures reported')
    return list(items.values()), source
