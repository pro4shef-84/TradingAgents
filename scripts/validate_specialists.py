"""Causal fixed-horizon specialist observations against matched-date long SPY."""
import argparse
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from tradingagents.technical_scan import validate_bars
from tradingagents.technique_panel import evaluate_stock


def date(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError('Expected a timezone-free session date')
    return result


def summarize(observations):
    groups = defaultdict(list)
    for row in observations:
        groups[(row['split'], row['rule_id'], row['direction'])].append(row)
    result = []
    for (split, rule, direction), rows in sorted(groups.items()):
        paired = [r for r in rows if r['excess_return'] is not None]
        result.append(dict(split=split, rule_id=rule, direction=direction,
            observations=len(rows), symbols=len({r['ticker'] for r in rows}),
            entry_dates=len({r['entry_at'] for r in rows}),
            mean_net_return=sum(r['net_return'] for r in rows)/len(rows),
            positive_fraction=sum(r['net_return'] > 0 for r in rows)/len(rows),
            benchmark_pairs=len(paired),
            mean_spy_return=sum(r['spy_return'] for r in paired)/len(paired) if paired else None,
            mean_excess_return=sum(r['excess_return'] for r in paired)/len(paired) if paired else None))
    return result


def evaluate_history(data, formation_end, start, end, holding_bars=5, cost_bps=20,
                     benchmark=None, progress=None):
    start, end, formation_end = map(date, (start, end, formation_end))
    if not start <= formation_end < end:
        raise ValueError('Expected start <= formation_end < end')
    if isinstance(holding_bars, bool) or not isinstance(holding_bars, int) or holding_bars < 1:
        raise ValueError('Holding bars must be a positive integer')
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('Costs must be finite and nonnegative')
    def bounded(frame):
        if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None
                or frame.index.hasnans or frame.index.has_duplicates
                or not frame.index.is_monotonic_increasing):
            raise ValueError('Expected unique, increasing timezone-free session dates')
        return validate_bars(frame.loc[:end])
    frames = {ticker: bounded(frame) for ticker, frame in data.items()}
    spy = bounded(benchmark) if benchmark is not None else frames.get('SPY')
    observations = []
    excluded_boundary = 0
    for ticker, frame in frames.items():
        # Retain all earlier bars as warm-up; evaluate no bars beyond end.
        frame = frame.loc[:end]
        for i, signal_date in enumerate(frame.index):
            if signal_date < start:
                continue
            prefix = frame.iloc[:i+1]
            kwargs = {'asof': str(signal_date.date())}
            if benchmark is not None:
                kwargs['benchmark'] = spy.loc[:signal_date]
            panel = evaluate_stock(prefix, **kwargs)
            if i + holding_bars >= len(frame):
                continue
            entry_date, exit_date = frame.index[i+1], frame.index[i+holding_bars]
            if signal_date <= formation_end < exit_date:
                excluded_boundary += 1
                continue
            entry = float(frame.Open.iloc[i+1])
            exit_price = float(frame.Close.iloc[i+holding_bars])
            spy_return = None
            if spy is not None and entry_date in spy.index and exit_date in spy.index:
                # Apply the same round-trip cost to the comparator.
                spy_return = float(spy.loc[exit_date, 'Close']/spy.loc[entry_date, 'Open']-1)-cost_bps/10000
            for agent in panel['agents']:
                if (agent['id'] == 50 or agent['status'] != 'matched'
                        or agent.get('direction') not in ('bullish', 'bearish')):
                    continue
                sign = 1 if agent['direction'] == 'bullish' else -1
                net = sign*(exit_price/entry-1)-cost_bps/10000
                observations.append(dict(ticker=ticker, rule_id=agent['id'],
                    direction=agent['direction'], evidence=agent.get('evidence'),
                    signal_at=str(signal_date.date()), entry_at=str(entry_date.date()),
                    exit_at=str(exit_date.date()), entry_open=entry, exit_close=exit_price,
                    net_return=net, spy_return=spy_return,
                    excess_return=net-spy_return if spy_return is not None else None,
                    split='formation' if exit_date <= formation_end else 'evaluation'))
        if progress:
            progress(ticker, len(observations))
    summaries = summarize(observations)
    selected = {(r['rule_id'], r['direction']) for r in summaries
                if r['split'] == 'formation' and r['observations'] >= 20
                and r['benchmark_pairs'] == r['observations']
                and r['mean_net_return'] > 0 and r['mean_excess_return'] > 0}
    for summary in summaries:
        summary['selected_in_formation'] = (summary['rule_id'], summary['direction']) in selected
    return dict(observations=observations, summaries=summaries,
        parameters=dict(start=str(start.date()), end=str(end.date()),
            formation_end=str(formation_end.date()), holding_bars=holding_bars,
            round_trip_cost_bps=cost_bps), excluded_boundary_windows=excluded_boundary,
        limitations=dict(point_in_time_supplemental='absent', surviving_universe_bias=True,
            certified_untouched_holdout=False, independent_trades=False,
            portfolio_returns=False, pairs_excluded=True,
            short_borrow_and_locates='unmodeled; bearish returns are theoretical',
            shortlist_selection='Current shortlist membership is retrospective',
            selection='Formation-only groups: >=20 paired observations, positive net and excess means'))


def compare_cohorts(result):
    selected = {(r['rule_id'], r['direction']) for r in result['summaries']
                if r['split'] == 'formation' and r['selected_in_formation']}
    comparisons = []
    for name in ('shortlist', 'control'):
        members = set(result.get('protocol', {}).get(name, []))
        for selection in ('all_matches', 'formation_selected'):
            windows = {}
            for r in result['observations']:
                if r['split'] != 'evaluation' or r['ticker'] not in members:
                    continue
                if selection == 'formation_selected' and (r['rule_id'], r['direction']) not in selected:
                    continue
                # Multiple rules on the same directional window share one outcome.
                windows[(r['ticker'], r['signal_at'], r['direction'])] = r
            paired = [r for r in windows.values() if r['spy_return'] is not None]
            comparisons.append(dict(cohort=name, selection=selection,
                windows=len(windows), benchmark_pairs=len(paired),
                entry_dates=len({r['entry_at'] for r in windows.values()}),
                mean_net_return=sum(r['net_return'] for r in paired)/len(paired) if paired else None,
                mean_spy_return=sum(r['spy_return'] for r in paired)/len(paired) if paired else None,
                mean_excess_return=sum(r['excess_return'] for r in paired)/len(paired) if paired else None))
    return comparisons


def render_report(result):
    def pct(value):
        return 'n/a' if value is None else f'{value*100:.2f}%'
    rows = []
    for r in result['summaries']:
        if r['split'] != 'evaluation':
            continue
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in
            (r['rule_id'], r['direction'], r['selected_in_formation'], r['observations'],
             r['entry_dates'], pct(r['mean_net_return']), pct(r['mean_spy_return']),
             pct(r['mean_excess_return'])))+'</tr>')
    limitations = html.escape(json.dumps(result['limitations'], indent=2))
    comparisons = compare_cohorts(result)
    overview = ''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in
        (r['cohort'], r['selection'], r['windows'], r['entry_dates'],
         pct(r['mean_net_return']), pct(r['mean_spy_return']), pct(r['mean_excess_return'])))+'</tr>'
        for r in comparisons)
    parameters = html.escape(json.dumps(result['parameters'], indent=2))
    return ('<!doctype html><html><meta charset="utf-8"><title>Specialist validation</title>'
        '<style>body{font:16px system-ui;margin:40px;max-width:1200px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style>'
        '<h1>Chronological specialist evaluation</h1><p>Mean returns per signal window, '
        'after costs, against long SPY over identical dates. Overlapping signals are '
        'not independent trades or portfolio returns. No claim of statistical confidence.</p>'
        '<h2>Shortlist and comparison universe</h2><p>One observation per stock, signal date '
        'and direction: duplicate rule matches share one outcome. Long and theoretical '
        'short windows are included. This does not replay the daily shortlist ranking policy.</p>'
        '<table><tr><th>Cohort</th><th>Selection</th><th>Windows</th><th>Entry dates</th>'
        '<th>Net</th><th>SPY</th><th>Excess</th></tr>'+overview+'</table>'
        '<h2>Protocol dates and costs</h2><pre>'+parameters+'</pre>'
        '<h2>Rule results</h2><table><tr><th>Rule</th><th>Direction</th><th>Formation selected</th><th>Signals</th>'
        '<th>Entry dates</th><th>Net</th><th>SPY</th><th>Excess</th></tr>'+''.join(rows)+
        '</table><h2>Limits</h2><pre>'+limitations+'</pre></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (args.output/'validation.json').exists():
        parser.error('Output already exists; choose a new directory')
    frames = {}
    for ticker in sorted(set(protocol['symbols']+['SPY'])):
        path = args.prices/f'{ticker}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol['input_hashes'][ticker]:
            parser.error(f'Frozen input hash mismatch: {ticker}')
        frames[ticker] = pd.read_csv(path, index_col=0, parse_dates=True)
    for relative, expected in protocol['code_hashes'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            parser.error(f'Frozen evaluator hash mismatch: {relative}')
    data = {t: frames[t] for t in protocol['symbols']}
    result = evaluate_history(data, **{k: protocol[k] for k in
        ('formation_end', 'start', 'end', 'holding_bars', 'cost_bps')}, benchmark=frames['SPY'],
        progress=lambda t,n: print(f'{t}: {n} cumulative observations', flush=True))
    result['protocol'] = protocol
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['cohorts'] = {name: summarize([r for r in result['observations']
        if r['ticker'] in protocol[name]]) for name in ('shortlist', 'control')}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    pd.DataFrame(result['observations']).to_csv(args.output/'observations.csv', index=False)
    (args.output/'report.html').write_text(render_report(result))
    print(f'Saved {args.output}/report.html', flush=True)


if __name__ == '__main__':
    main()
