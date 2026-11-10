"""Bounded, read-only current news and authorized Discord observations.

Collection is driven by the trade automation scheduler. No social sentiment is inferred.
"""
import os
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import dotenv_values
from tradingagents.pilot.providers import alpaca_get
from tradingagents.pilot.screen import utcnow

ROOT = Path(__file__).resolve().parents[2]


def safe_url(value):
    value = str(value or '')
    u = urlparse(value)
    return value if u.scheme in ('https', 'http') and u.hostname and not u.username else ''


def in_window(value, start, end):
    try:
        t = pd.Timestamp(value)
        return t.tzinfo is not None and start <= t <= end
    except (ValueError, TypeError):
        return False


def news(tickers, now=None, since=None):
    now = pd.Timestamp(now or utcnow()); start = pd.Timestamp(since) if since else now - pd.Timedelta(days=7)
    params = dict(symbols=','.join(t.replace('-', '.') for t in tickers), start=start.isoformat(),
                  end=now.isoformat(), limit=50, sort='desc', include_content='false')
    items = {}; tokens = set(); truncated = False
    for _ in range(10):
        data = alpaca_get('/v1beta1/news', params)
        for a in data.get('news', []):
            symbols = sorted(set(tickers) & {s.replace('.', '-') for s in a.get('symbols', [])})
            if not symbols or not in_window(a.get('created_at'), start, now): continue
            id = 'alpaca:' + str(a['id'])
            items[id] = dict(id=id, kind='news', source=a.get('source') or 'Alpaca news',
                title=str(a.get('headline', ''))[:1000], url=safe_url(a.get('url')),
                published_at=a['created_at'], updated_at=a.get('updated_at'),
                fetched_at=now.isoformat(), symbols=symbols)
        token = data.get('next_page_token')
        if not token: break
        if token in tokens: raise ValueError('Repeated news pagination token')
        tokens.add(token); params['page_token'] = token
    else: truncated = True
    return list(items.values()), dict(status='connected', fetched_at=now.isoformat(),
        truncated=truncated, coverage='Requested time window; at most 500 articles per batch', window_start=start.isoformat())


def discord(tickers, now=None, max_pages=3):
    if not isinstance(max_pages,int) or not 1<=max_pages<=10:raise ValueError('Invalid Discord page budget')
    now = pd.Timestamp(now or utcnow()); start = now - pd.Timedelta(days=7)
    local = dotenv_values(ROOT / '.env')
    token = os.environ.get('DISCORD_BOT_TOKEN') or local.get('DISCORD_BOT_TOKEN')
    channels = os.environ.get('DISCORD_CHANNEL_IDS') or local.get('DISCORD_CHANNEL_IDS') or ''
    channels = [c.strip() for c in channels.split(',') if c.strip()]
    if not token or not channels:
        return [], dict(status='not_connected', detail='Needs bot token and authorized channel IDs; Slack webhook cannot read Discord')
    if len(channels) > 5 or any(not c.isdigit() for c in channels):
        raise ValueError('Configure one to five numeric Discord channel IDs')
    items = []; capped = False; empty_content = False; message_count = 0; page_count = 0; seen_messages = set()
    for channel in channels:
        messages = []; before = None; cursors = set()
        for page_number in range(max_pages):
            params = {'limit':100}
            if before: params['before'] = before
            try:
                r = requests.get(f'https://discord.com/api/v10/channels/{channel}/messages',
                    params=params, headers={'Authorization':'Bot '+token}, timeout=20)
            except requests.RequestException:
                raise ValueError('Discord network request failed') from None
            if r.status_code != 200: raise ValueError(f'Discord HTTP {r.status_code}; check bot channel access/content intent or rate limit')
            page = r.json(); page_count += 1
            if not isinstance(page,list) or len(page)>100: raise ValueError('Malformed Discord messages response')
            if any(not isinstance(a,dict) or not str(a.get('id','')).isdigit() for a in page):raise ValueError('Invalid Discord message identifier')
            messages.extend(page);message_count += len(page)
            if len(page)<100: break
            before = min((str(a['id']) for a in page),key=int)
            if before in cursors:raise ValueError('Repeated Discord pagination cursor')
            cursors.add(before)
            if page_number==max_pages-1:capped = True
        try:
            channel_response = requests.get(f'https://discord.com/api/v10/channels/{channel}',
                headers={'Authorization':'Bot '+token}, timeout=20)
        except requests.RequestException:
            raise ValueError('Discord channel lookup failed') from None
        if channel_response.status_code != 200: raise ValueError(f'Discord channel lookup HTTP {channel_response.status_code}')
        guild = str(channel_response.json().get('guild_id') or '@me')
        if guild != '@me' and not guild.isdigit(): raise ValueError('Invalid Discord guild identifier')
        for a in messages:
            if a['id'] in seen_messages:continue
            seen_messages.add(a['id'])
            content = a.get('content') or ''
            empty_content |= not content and not a.get('embeds')
            # Explicit cashtags avoid common-word ticker false positives.
            symbols = sorted(set(tickers) & set(re.findall(r'\$([A-Z][A-Z0-9.-]{0,9})\b', content)))
            if not symbols or not in_window(a.get('timestamp'), start, now): continue
            available = a.get('edited_timestamp') or a['timestamp']
            if not in_window(available,start,now):continue
            items.append(dict(id='discord:'+a['id'],kind='social',source='Discord · '+channel,
                title=content[:1500],url=f'https://discord.com/channels/{guild}/{channel}/{a["id"]}',
                published_at=a['timestamp'],updated_at=a.get('edited_timestamp'),available_at=available,
                fetched_at=now.isoformat(),symbols=symbols))
    return items, dict(status='connected', fetched_at=now.isoformat(), truncated=capped,
        possible_missing_content=empty_content,
        message_count=message_count,page_count=page_count,max_pages_per_channel=max_pages,
        coverage=f'At most {max_pages*100} messages/channel; explicit cashtags only; images and embeds not analyzed')
