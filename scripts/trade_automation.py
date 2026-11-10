"""Install, inspect or run local trade-plan and Slack monitoring automation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tradingagents.pilot import automation as a
from tradingagents.pilot.daily import save_json
from tradingagents.pilot.operations import slack_webhook
from tradingagents.pilot.providers import alpaca_get
from tradingagents.pilot.screen import utcnow
from tradingagents.pilot.signal_monitor import new_state


def unload():
    target=f'gui/{os.getuid()}/{a.LABEL}'
    subprocess.run(['launchctl','bootout',target],capture_output=True,check=False)
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        if subprocess.run(['launchctl','print',target],capture_output=True).returncode:return
        time.sleep(.1)
    raise ValueError('LaunchAgent has not finished unloading; automation is paused in configuration')


def status():
    path=Path.home()/'Library/LaunchAgents'/f'{a.LABEL}.plist'
    result=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/{a.LABEL}'],capture_output=True,text=True)
    state=a.load(a.STATE,{})
    return dict(installed=path.exists(),loaded=result.returncode==0,
                config=a.load(a.CONFIG,{}),heartbeat={k:state.get(k) for k in ('last_tick_at','last_tick_completed_at','last_error','pid')},
                pending_delivery=sum(i['status']=='pending' for i in state.get('outbox',{}).values()),
                uncertain_delivery=sum(i['status'] in ('sending','uncertain') for i in state.get('outbox',{}).values()),
                signals={k:s['status'] for k,s in state.get('signals',{}).items()})


def install(seed=None):
    a.BASE.mkdir(parents=True,exist_ok=True)
    hook=slack_webhook()
    if not hook:raise ValueError('Slack webhook not configured')
    feed=None;probes={}
    for candidate in ('sip','iex'):
        try:
            response=alpaca_get('/v2/stocks/quotes/latest',dict(symbols='SPY',feed=candidate))
            if response.get('quotes',{}).get('SPY'):
                feed=candidate;probes[candidate]='authorized';break
        except Exception as exc:probes[candidate]=type(exc).__name__
    if not feed:raise ValueError('No authorized quote feed; automation not installed')
    label='SIP consolidated' if feed=='sip' else 'IEX single-exchange'
    config=dict(enabled=True,installed_at=utcnow(),feed=feed,feed_label=label,
        feed_authorization=probes,poll_seconds=60,freshness_seconds=180,
        destination_sha256=hashlib.sha256(hook.encode()).hexdigest(),
        schedule_timezone='America/Denver',morning='06:00',premarket_update='06:45',
        monitoring='XNYS regular session; entries after first completed 15-minute candle',
        economic_events=[dict(session='2026-11-10',kind='CPI',display='CPI 8:30 a.m. ET / 6:30 a.m. Mountain; wait for the regular-session confirmation')],
        broker_orders_enabled=False)
    save_json(a.CONFIG,config)
    state=a.load(a.STATE,dict(jobs={},signals={},outbox={}))
    if seed:
        data=json.loads(seed.read_text())
        config['reviewed_plans']={p['ticker']:data['prepared_at'] for p in data['plans']}
        save_json(a.CONFIG,config)
        for plan in data['plans']:
            plan={**plan,'trade_session':data['trade_session'],'trigger':plan.get('confirmation','confirmation and retest')}
            state['signals'].setdefault(a.key(plan),new_state(plan))
    save_json(a.STATE,state)
    logs=a.BASE/'logs';logs.mkdir(exist_ok=True)
    path=Path.home()/'Library/LaunchAgents'/f'{a.LABEL}.plist';path.parent.mkdir(parents=True,exist_ok=True)
    unload()
    spec=dict(Label=a.LABEL,ProgramArguments=['/usr/bin/caffeinate','-i','-s',str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/trade_automation.py'),'serve'],
        WorkingDirectory=str(ROOT),RunAtLoad=True,KeepAlive=True,ThrottleInterval=30,
        StandardOutPath=str(logs/'service.out.log'),StandardErrorPath=str(logs/'service.err.log'),
        EnvironmentVariables=dict(PYTHONUNBUFFERED='1',PATH='/usr/bin:/bin:/usr/sbin:/sbin'))
    path.write_bytes(plistlib.dumps(spec))
    result=subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(path)],capture_output=True)
    if result.returncode:raise ValueError(f'LaunchAgent bootstrap failed ({result.returncode})')
    return status()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['install','status','pause','resume','tick','serve'])
    parser.add_argument('--seed',type=Path)
    args=parser.parse_args()
    if args.command=='serve':a.serve();return
    if args.command=='install':result=install(args.seed)
    elif args.command=='status':result=status()
    elif args.command in ('pause','resume'):
        config=a.load(a.CONFIG,{});config['enabled']=args.command=='resume';save_json(a.CONFIG,config)
        if args.command=='pause':unload()
        elif not status()['loaded']:
            path=Path.home()/'Library/LaunchAgents'/f'{a.LABEL}.plist'
            command=subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(path)],capture_output=True)
            if command.returncode:raise ValueError('Could not resume LaunchAgent')
        result=status()
    else:result=a.tick()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
