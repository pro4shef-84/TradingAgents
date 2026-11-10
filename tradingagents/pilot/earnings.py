"""Bounded, fail-closed Yahoo earnings-calendar lookup for candidate plans."""
import multiprocessing as mp

import pandas as pd


def _date(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('Missing earnings date')
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('America/New_York')
    return stamp.date()


def _lookup(ticker, session, pipe):
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        start = _date(session)
        try:
            calendar = stock.calendar
            values = calendar.get('Earnings Date', []) if isinstance(calendar, dict) else []
            dates = sorted({_date(value) for value in values})
        except Exception:
            dates = []
        if not dates or max(dates) < start:
            history = stock.get_earnings_dates(limit=12)
            upcoming = sorted({_date(value) for value in history.index if _date(value) >= start}) if history is not None else []
            dates = upcoming[:1]
        pipe.send([value.isoformat() for value in dates])
    except Exception:
        pipe.send([])
    finally:
        pipe.close()


def earnings_dates(ticker, session, timeout=15):
    """Return the scheduled date/range, or [] for failure/unknown; never infer a date."""
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_lookup, args=(ticker, session, sender), daemon=True)
    try:
        worker.start()
        sender.close()
        return receiver.recv() if receiver.poll(timeout) else []
    except (EOFError, OSError):
        return []
    finally:
        if worker.pid is not None:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=2)
        receiver.close()
        sender.close()


def earnings_rejection(dates, session, calendar):
    if not dates:
        return 'earnings_date_unknown'
    try:
        first = _date(session)
        last = _date(calendar.next_session(calendar.next_session(session)))
        values = [_date(value) for value in dates]
    except (ValueError, TypeError):
        return 'earnings_date_unknown'
    if max(values) < first:
        return 'earnings_date_unknown'
    # Yahoo can publish a range. Any overlap with the protected window rejects it.
    if min(values) <= last and max(values) >= first:
        return 'earnings_within_2_sessions'
    return None
