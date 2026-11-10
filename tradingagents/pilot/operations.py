"""Slack destination configuration and validated supplemental-data imports."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from dotenv import dotenv_values

from tradingagents.pilot.daily import save_json

ROOT=Path(__file__).resolve().parents[2]


def slack_webhook():
    value=os.environ.get('INVESTOR_SLACK_WEBHOOK') or dotenv_values(ROOT/'.env').get('INVESTOR_SLACK_WEBHOOK')
    if not value:return None
    parsed=urlparse(value)
    if parsed.scheme!='https' or parsed.hostname!='hooks.slack.com' or not parsed.path.startswith('/services/'):
        raise ValueError('Invalid Slack webhook configuration')
    return value


def import_data(root, source, kind):
    from tradingagents.pilot.market_inputs import import_records
    root=Path(root)
    if kind=='pairs':
        source=Path(source)
        if source.stat().st_size>10_000_000:raise ValueError('Import exceeds 10MB budget')
        batch=json.loads(source.read_text())
        if not isinstance(batch,dict) or len(batch)>20:raise ValueError('Pairs import requires at most 20 per-symbol evaluator envelopes')
        # Evaluate metadata/records with the same causal validator used by the actual rule.
        from tradingagents.specialist_rules import dataset
        cutoff=pd.Timestamp.now(tz='UTC')
        for symbol,ds in batch.items():
            if not isinstance(ds,dict) or ds.get('symbol')!=symbol:raise ValueError('Pair symbol mismatch')
            dataset({'symbol':symbol,'pairs':ds},'pairs',cutoff)
            if ds.get('adjustment')!='adjusted' or len(ds.get('pair',[]))!=2:raise ValueError('Explicit adjusted pair required')
    else:
        batch=import_records(source,kind)
    path=root/'results/pilot/imports'/f'{kind}.json'
    save_json(path,batch)
    return dict(status='imported',kind=kind,path=str(path),live_connection=False)
