"""Versioned causal research proxies; see docs/technique-rules.md.

No profitability, investor identity or live entitlement is inferred here.
"""
from collections import defaultdict
from urllib.parse import urlparse
import math
import json
import numpy as np
import pandas as pd
import exchange_calendars as xcals

_CALENDAR = xcals.get_calendar("XNYS")
from tradingagents.technical_scan import pivots

VERSION = 'specialist_rules_v1'
DAILY_IDS = (5, 6, 12, 14, 25, 39)
SUPPLEMENTAL_IDS = (15, 30, 38, 42, 43, 44, 46, 47, 50)


def result(status='no_match', reason=None):
    return dict(status=status, direction='neutral' if status in ('no_match','context') else None,
                fit_score=0. if status=='no_match' else None, evidence=reason or [], events=[], rule_version=VERSION)


def signal(direction, evidence, level, invalidation, start, end, status='matched', **meta):
    evidence=dict(evidence, rule_version=VERSION, observed_window=[str(start),str(end)],
                  confirmation_time=str(end) if status=='matched' else None, observed_at=str(end), trigger=float(level), invalidation=float(invalidation), **meta)
    event=dict(direction=direction, evidence=evidence, level=float(level),status=status,
               invalidation=float(invalidation),timeframe=meta.get('timeframe','daily'))
    return dict(status=status,direction=direction,fit_score=70. if status=='matched' else 40. if status=='watch' else 20.,
                evidence=[evidence],events=[event],rule_version=VERSION)


def daily_rules(d):
    out={i:result() for i in DAILY_IDS}
    atr=float(d.atr.iloc[-1]); c=d.Close.to_numpy(); h=d.High.to_numpy(); l=d.Low.to_numpy()
    if not math.isfinite(atr) or atr<=0:
        return {i:result('invalid_data','Positive ATR required') for i in DAILY_IDS}
    now=float(c[-1]); end=d.index[-1]
    def emit(i,side,level,stop,params,window,status='matched'):
        out[i]=signal(side,params,level,stop,d.index[-window],end,status)
    cup=c[-41:-1]; means=[float(x.mean()) for x in np.split(cup,4)]
    if (abs(means[0]-means[3])<=2*atr and min(means[0],means[3])-max(means[1:3])>=atr
            and np.polyfit(range(20),cup[:20],1)[0]<0<np.polyfit(range(20),cup[20:],1)[0]):
        rim=float(max(cup)); emit(5,'bullish',rim,min(cup),dict(window=40,outer_tolerance_atr=2,middle_depth_atr=1),41,'matched' if now>rim else 'watch')
    widths=[float(max(x)-min(y)) for x,y in zip(np.split(h[-33:-1],4),np.split(l[-33:-1],4))]
    if widths[0]>0 and widths[1]>=1.5*widths[0] and widths[3]<=.75*widths[1] and widths[2]<=widths[1]:
        top,bottom=float(max(h[-9:-1])),float(min(l[-9:-1]))
        side='bullish' if now>top else 'bearish' if now<bottom else 'neutral'
        emit(6,side,top if side!='bearish' else bottom,bottom if side!='bearish' else top,
             dict(window=32,block_widths=widths,proxy='range expansion/contraction'),33,'watch' if side=='neutral' else 'matched')
    cup=c[-49:-9]; handle=c[-9:-1]
    left,right=float(cup[:5].mean()),float(cup[-5:].mean()); rim=max(left,right)
    depth=rim-float(cup[15:25].mean()); handle_depth=rim-float(min(handle))
    if (abs(left-right)<=atr and depth>=2*atr and .25*atr<=handle_depth<=.5*depth
            and min(handle)>rim-.5*depth and np.polyfit(range(20),cup[:20],1)[0]<0<np.polyfit(range(20),cup[20:],1)[0]):
        emit(12,'bullish',rim,min(handle),dict(cup_window=40,handle_window=8,depth_atr=depth/atr),49,'matched' if now>rim else 'watch')
    vol=float(d.Volume.iloc[-21:-1].mean())
    for sign,side in ((1,'bullish'),(-1,'bearish')):
        x=sign*c; upper=h if sign==1 else -l; lower=l if sign==1 else -h
        if (lower[-1]-upper[-2]>=.25*atr and x[-2]-x[-12]>=3*atr and d.Volume.iloc[-1]>=1.5*vol):
            emit(14,side,sign*upper[-2],sign*lower[-1],dict(gap_atr=float((lower[-1]-upper[-2])/atr),impulse_bars=10,volume_multiple=1.5),21)
        candidates=[p for p in pivots(-lower) if len(d)-21<=p<=len(d)-4]
        if candidates:
            p=candidates[-1]; zone=lower[p]
            if zone-.25*atr<=lower[-1]<=zone+.25*atr and x[-1]>=zone+.5*atr:
                emit(25,side,sign*(zone+.25*atr),sign*(zone-.25*atr),dict(pivot_time=str(d.index[p]),zone_width_atr=.5,proxy='price-zone reaction',historical_name='Order blocks',ownership='unknown'),21)
        boundary=float(min(lower[-21:-1])); top=float(max(upper[-21:-1]))
        if (top-boundary<=8*atr and lower[-1]<=boundary-.1*atr and x[-1]>=boundary+.25*atr and d.Volume.iloc[-1]>=1.5*vol):
            emit(39,side,sign*boundary,sign*lower[-1],dict(range_bars=20,probe_atr=.1,reclaim_atr=.25,volume_multiple=1.5,proxy='spring/upthrust'),21)
    return out


class InputError(ValueError):
    def __init__(self, reason, status='invalid_data'):
        self.status=status
        super().__init__(reason)


def timestamp(value):
    try:
        t=pd.Timestamp(value)
        if t.tzinfo is None or pd.isna(t):
            raise ValueError()
        return t.tz_convert('UTC')
    except (ValueError,TypeError,OverflowError):
        raise InputError('Timezone-aware valid timestamp required') from None


def number(value, positive=False):
    try:
        v=float(value)
        if not math.isfinite(v) or (positive and v<=0):
            raise ValueError()
        return v
    except (ValueError,TypeError,OverflowError):
        raise InputError('Finite positive numeric input required' if positive else 'Finite numeric input required') from None


def dataset(s, key, cutoff, complete=True):
    ds=s.get(key)
    if not isinstance(ds,dict):
        raise InputError('Missing '+key+' observations','missing_evidence' if key=='events' else 'missing_data')
    required=('provider','feed','symbol','retrieved_at','adjustment','coverage','records')
    if any(k not in ds for k in required):
        raise InputError('Dataset metadata missing: '+key)
    timestamp(ds['retrieved_at'])
    if s.get('symbol') and ds['symbol']!=s['symbol']:
        raise InputError('Dataset symbol mismatch')
    if not isinstance(ds['coverage'],dict) or not isinstance(ds['records'],list):
        raise InputError('Invalid coverage/records contract')
    try:
        json.dumps(ds['coverage'],allow_nan=False)
    except (TypeError,ValueError):
        raise InputError('Coverage metadata must be finite JSON') from None
    if any(not isinstance(ds[key],str) or not ds[key] for key in ('provider','feed','symbol','adjustment')):
        raise InputError('Provider/feed/symbol/adjustment must be nonempty strings')
    if complete and ds['coverage'].get('complete') is not True:
        raise InputError('Incomplete '+key+' coverage','missing_data')
    records=[]
    for record in ds['records']:
        if not isinstance(record,dict):
            raise InputError('Record must be an object')
        t=timestamp(record.get('timestamp',record.get('event_at',record.get('event_time'))))
        available=timestamp(record.get('available_at'))
        if record.get('updated_at'):
            available=max(available,timestamp(record['updated_at']))
        if t>cutoff or available>cutoff:
            continue
        if available<t:
            raise InputError('Availability precedes event')
        if record.get('symbol',ds['symbol'])!=ds['symbol']:
            raise InputError('Record symbol mismatch')
        records.append(dict(record,_t=t,_a=available))
    records.sort(key=lambda r:(r['_t'],r['_a']))
    if not records:
        raise InputError('No available '+key+' observations','missing_evidence' if key=='events' else 'missing_data')
    return ds,records


def session_bounds(day, ds):
    if not _CALENDAR.is_session(day):
        raise InputError('Nontrading XNYS session')
    opening=_CALENDAR.session_open(day)
    closing=_CALENDAR.session_close(day)
    declared=ds.get('session_close') if day==ds.get('session') else None
    if declared is not None and timestamp(declared)!=closing:
        raise InputError('Declared session close disagrees with XNYS calendar')
    return opening,closing


def intraday(s,cutoff):
    ds,records=dataset(s,'intraday',cutoff)
    sessions=defaultdict(list)
    seen=set()
    for r in records:
        t=r['_t'].tz_convert('America/New_York'); day=str(t.date())
        duration=number(r.get('interval_minutes'),True)
        if duration not in (1,5,10,15,30):
            raise InputError('Unsupported interval')
        finish=r['_t']+pd.Timedelta(minutes=duration)
        if finish>cutoff:
            continue
        if r['_a']<finish:
            raise InputError('Bar available before completion')
        if r['_t'] in seen:
            raise InputError('Duplicate intraday bar')
        seen.add(r['_t'])
        values=[number(r.get(k),True) for k in ('open','high','low','close')]
        o,h,l,c=values; v=number(r.get('volume'),True)
        if not l<=min(o,c)<=max(o,c)<=h:
            raise InputError('Invalid intraday OHLC')
        if r.get('session',day)!=day:
            raise InputError('Session/date mismatch')
        opening,closing=session_bounds(day,ds)
        if t<opening or finish>closing:
            continue
        sessions[day].append(dict(r,open=o,high=h,low=l,close=c,volume=v,_finish=finish))
    if not sessions:
        raise InputError('No completed regular-session bars','missing_data')
    for day,rs in sessions.items():
        opening,_=session_bounds(day,ds)
        step=rs[0]['interval_minutes']
        expected=pd.date_range(opening,rs[-1]['_t'],freq=pd.Timedelta(minutes=step))
        if any(r['interval_minutes']!=step for r in rs) or list(expected)!=[r['_t'] for r in rs]:
            raise InputError('Missing session opening/interval coverage','missing_data')
    return ds,sessions


def supplemental_rules(d,s):
    out={i:result('missing_evidence' if i==47 else 'missing_data','Required supplemental evidence unavailable') for i in SUPPLEMENTAL_IDS}
    if not s:
        return out
    try:
        cutoff=timestamp(s.get('cutoff'))
    except InputError as exc:
        return {i:result(exc.status,str(exc)) for i in SUPPLEMENTAL_IDS}
    def emit(i,side,level,stop,params,rs,ds,status='matched'):
        out[i]=signal(side,params,level,stop,rs[0]['_t'].isoformat(),rs[-1]['_a'].isoformat(),status,
                      timeframe='intraday' if i!=50 else 'pair_spread',provider=ds['provider'],feed=ds['feed'],
                      coverage=ds['coverage'],cutoff=cutoff.isoformat())
    def run(ids,func):
        try:
            func()
        except (InputError,KeyError,TypeError,ValueError,IndexError) as exc:
            for i in ids:
                out[i]=result(getattr(exc,'status','invalid_data'),str(exc))
    def bar_rules():
        ds,sessions=intraday(s,cutoff); rs=sessions[max(sessions)]; last=rs[-1]; out[15]=result();out[30]=result();out[46]=result()
        opening=rs[0]['_t']; opening_end=opening+pd.Timedelta(minutes=30)
        first=[r for r in rs if r['_finish']<=opening_end]; after=[r for r in rs if r['_t']>=opening_end]
        if not first or first[-1]['_finish']!=opening_end or not after:
            out[15]=result('missing_data','Opening 30 minutes plus completed subsequent bar required')
            out[46]=result('missing_data','First 30-minute confirmation unavailable')
        else:
            top=max(r['high'] for r in first); bottom=min(r['low'] for r in first); previous=after[-2]['close'] if len(after)>1 else first[-1]['close']
            for side,matched,level,stop in [('bullish',last['close']>top>=previous,top,bottom),('bearish',last['close']<bottom<=previous,bottom,top)]:
                if matched:
                    emit(15,side,level,stop,dict(opening_minutes=30),rs,ds)
        def gap_rule():
            if not first or first[-1]['_finish']!=opening_end:
                out[46]=result('missing_data','First 30-minute confirmation unavailable')
                return
            expected=str(_CALENDAR.previous_session(max(sessions)).date())
            reference=s.get('previous_close')
            if reference is not None:
                if not isinstance(reference,dict):
                    raise InputError('Previous close must be a sourced reference')
                ref_date=str(pd.Timestamp(reference.get('asof')).date())
                if ref_date!=expected:
                    raise InputError('Previous close must reference immediately previous XNYS session')
                if not reference.get('provider') or not reference.get('feed'):
                    raise InputError('Previous close provenance required')
                prior=number(reference.get('price'),True)
                adjustment=reference.get('adjustment')
            else:
                prior_rows=d.loc[[str(index.date())<max(sessions) for index in d.index]]
                if prior_rows.empty or str(prior_rows.index[-1].date())!=expected:
                    out[46]=result('missing_data','Previous XNYS session close unavailable')
                    return
                prior=float(prior_rows.Close.iloc[-1]);ref_date=str(prior_rows.index[-1].date())
                adjustment=d.attrs.get('adjustment')
            if not adjustment or adjustment!=ds['adjustment']:
                out[46]=result('missing_data','Unknown or mismatched daily/intraday adjustment; comparable previous close required')
                return
            op=rs[0]['open'];gap=op/prior-1;out[46]=result()
            if abs(gap)>=.01 and (last['close']/op-1)*np.sign(gap)>=.005:
                emit(46,'bullish' if gap>0 else 'bearish',op,prior,dict(gap_return=gap,previous_close_time=ref_date,premarket_observed=bool(ds.get('premarket_observed',False)),adjustment_comparability='declared matching',previous_close_source=reference if reference is not None else dict(provider='daily_bars',adjustment=adjustment)),rs,ds)
        run((46,),gap_rule)
        if len(rs)>=2:
            approximate=any(r.get('vwap') is None for r in rs)
            prices=[(r['high']+r['low']+r['close'])/3 if approximate else number(r['vwap'],True) for r in rs]
            volumes=np.array([r['volume'] for r in rs]); cumulative=np.cumsum(np.array(prices)*volumes)/np.cumsum(volumes)
            a=rs[-2]['close']-cumulative[-2]; b=last['close']-cumulative[-1]
            if b>0>=a or b<0<=a:
                emit(30,'bullish' if b>0 else 'bearish',cumulative[-1],min(r['low'] for r in rs) if b>0 else max(r['high'] for r in rs),dict(approximate=approximate,method='HLC3 volume weighting approximation' if approximate else 'provider bar VWAP volume weighting'),rs,ds)
        out[44]=result('missing_data','Two complete sessions required for Market Profile')
        if len(sessions)>=2:
            keys=sorted(sessions)[-2:]; areas=[]; bin_size=number(ds.get('bin_size',.25),True)
            for key in keys:
                rr=sessions[key]; blocks=defaultdict(list)
                _,close=session_bounds(key,ds)
                if rr[-1]['_finish']!=close:
                    return
                for r in rr:
                    slot=int((r['_t']-rr[0]['_t']).total_seconds()//1800);blocks[slot].append(r)
                counts=defaultdict(int)
                for block in blocks.values():
                    if sum(r['interval_minutes'] for r in block)!=30:
                        raise InputError('Incomplete 30-minute TPO block','missing_data')
                    typical=(max(r['high'] for r in block)+min(r['low'] for r in block)+block[-1]['close'])/3
                    counts[math.floor(typical/bin_size)]+=1
                selected=[];total=0
                for k,count in sorted(counts.items(),key=lambda item:(-item[1],item[0])):
                    selected.append(k);total+=count
                    if total>=.7*sum(counts.values()):break
                areas.append((min(selected)*bin_size,(max(selected)+1)*bin_size))
            out[44]=result()
            if all(b>a for a,b in zip(areas[0],areas[1])) or all(b<a for a,b in zip(areas[0],areas[1])):
                side='bullish' if areas[1][0]>areas[0][0] else 'bearish'
                emit(44,side,areas[1][1] if side=='bullish' else areas[1][0],areas[1][0] if side=='bullish' else areas[1][1],dict(value_areas=areas,bin_size=bin_size,method='30-minute HLC3 time observations; 70% descending count bins'),rs,ds)
    run((15,30,44,46),bar_rules)
    def profile():
        ds,rs=dataset(s,'trades',cutoff); eligible=clean_trades(rs,ds['provider']); bin_size=number(ds.get('bin_size',.25),True)
        if not eligible:raise InputError('No eligible trades','missing_data')
        bins=defaultdict(float)
        for r in eligible:bins[math.floor(r['price']/bin_size)]+=r['size']
        poc=min(bins,key=lambda k:(-bins[k],k))*bin_size;last=eligible[-1]['price'];out[38]=result()
        if abs(last-poc)>=bin_size:
            emit(38,'bullish' if last>poc else 'bearish',poc,poc,dict(poc=poc,bin_size=bin_size,method='trade volume at price',eligible_trades=len(eligible)),eligible,ds)
    run((38,),profile)
    def tape():
        ds,rs=dataset(s,'trades',cutoff,complete=False);quote_ds,quotes=dataset(s,'quotes',cutoff,complete=False);rs=clean_trades(rs,ds['provider'])
        if len(rs)<10:raise InputError('At least ten eligible trades required','missing_data')
        for q in quotes:
            q['bid']=number(q.get('bid'),True);q['ask']=number(q.get('ask'),True)
            if q['bid']>q['ask']:raise InputError('Crossed quote')
        buy=sell=classified=0.;total=sum(r['size'] for r in rs)
        for r in rs:
            prior=[q for q in quotes if q['_t']<=r['_t'] and q['_a']<=r['_t'] and (r['_t']-q['_t']).total_seconds()<=2]
            if not prior:continue
            q=prior[-1];mid=(q['bid']+q['ask'])/2
            if r['price']==mid:continue
            classified+=r['size']
            if r['price']>mid:buy+=r['size']
            else:sell+=r['size']
        if classified/total<.8:raise InputError('Less than 80% volume aligned to fresh prior quotes','missing_data')
        out[42]=result()
        if max(buy,sell)/classified>=.65:
            emit(42,'bullish' if buy>sell else 'bearish',rs[-1]['price'],rs[0]['price'],dict(buy_volume=buy,sell_volume=sell,quote_coverage=quote_ds['coverage'],classified_fraction=classified/total,stale_quote_seconds=2,limitations='Observed feed only; midpoint prints unclassified; conditions/corrections excluded'),rs,ds)
    run((42,),tape)
    def book():
        ds,rs=dataset(s,'book',cutoff,complete=False);seq=None;state=None
        for r in rs:
            current=r.get('sequence')
            if not isinstance(current,int) or isinstance(current,bool):raise InputError('Integer book sequence required')
            if r.get('type')=='snapshot':seq=current;state=r
            elif r.get('type')=='update':
                if seq is None or current!=seq+1:seq=None;state=None
                else:seq=current;state=r
            else:raise InputError('Book type snapshot/update required')
        if state is None:raise InputError('Book sequence gap; fresh snapshot required','missing_data')
        sides=[]
        for name,reverse in [('bids',True),('asks',False)]:
            levels=state.get(name,[])
            if not levels:raise InputError('Empty book side')
            levels=[(number(p,True),number(v,True)) for p,v in levels]
            if len({p for p,_ in levels})!=len(levels):raise InputError('Duplicate book level')
            sides.append(sorted(levels,reverse=reverse)[:5])
        if sides[0][0][0]>=sides[1][0][0]:raise InputError('Crossed/locked book')
        bid,ask=[sum(v for _,v in levels) for levels in sides];imbalance=(bid-ask)/(bid+ask);out[43]=result()
        if abs(imbalance)>=.6:
            emit(43,'bullish' if imbalance>0 else 'bearish',sides[1][0][0] if imbalance>0 else sides[0][0][0],sides[0][0][0] if imbalance>0 else sides[1][0][0],dict(imbalance=imbalance,levels=5,sequence=seq,live_access='not_inferred'),rs,ds)
    run((43,),book)
    def catalyst():
        ds,events=dataset(s,'events',cutoff,complete=False);_,sessions=intraday(s,cutoff);rs=[r for day in sorted(sessions) for r in sessions[day]]
        verified=[e for e in events if e.get('verified') is True and urlparse(e.get('source_url','')).scheme in ('http','https') and urlparse(e.get('source_url','')).netloc and e.get('direction') in ('bullish','bearish')]
        if not verified:raise InputError('No verified source-linked catalyst','missing_evidence')
        out[47]=result('watch','Verified event awaiting postavailability price confirmation')
        for e in verified:
            event_time=timestamp(e.get('event_time'))
            if event_time>e['_a']:raise InputError('Event became available before event time')
            post=[r for r in rs if r['_t']>=e['_a']]
            if len(post)<2:continue
            move=post[-1]['close']/post[0]['open']-1;sign=1 if e['direction']=='bullish' else -1
            if sign*move>=.01:
                emit(47,e['direction'],post[0]['open'],min(r['low'] for r in post) if sign==1 else max(r['high'] for r in post),dict(source_url=e['source_url'],event_time=event_time.isoformat(),available_at=e['_a'].isoformat(),postavailability_return=move),post,ds)
    run((47,),catalyst)
    def pairs():
        ds,rs=dataset(s,'pairs',cutoff)
        if ds.get('adjustment')!='adjusted' or len(ds.get('pair',[]))!=2 or ds['pair'][0]!=ds['symbol'] or ds['pair'][0]==ds['pair'][1]:raise InputError('Explicit adjusted pair universe required')
        cost=number(ds.get('cost_bps'))
        if cost<0 or not isinstance(ds.get('borrow_assumption'),str) or not ds['borrow_assumption']:raise InputError('Costs and borrow assumption required')
        if len(rs)<65:raise InputError('60 formation and five trading observations required','insufficient_history')
        if len({r['_t'] for r in rs})!=len(rs):raise InputError('Duplicate paired timestamp')
        rs=rs[-65:];x=np.log([number(r.get('partner'),True) for r in rs]);y=np.log([number(r.get('stock'),True) for r in rs]);train_x=x[:60];train_y=y[:60]
        if np.std(train_x)<1e-8:raise InputError('Constant partner formation prices','missing_data')
        beta,alpha=np.polyfit(train_x,train_y,1);residual=train_y-alpha-beta*train_x;sd=float(np.std(residual,ddof=1))
        if sd<1e-8 or not .2<=beta<=5:raise InputError('Unstable/degenerate formation hedge','missing_data')
        betas=[]
        for lo,hi in ((0,30),(30,60)):
            if np.std(train_x[lo:hi])<1e-8:raise InputError('Degenerate formation half','missing_data')
            betas.append(float(np.polyfit(train_x[lo:hi],train_y[lo:hi],1)[0]));half=residual[lo:hi]
            rho=float(np.corrcoef(half[:-1],half[1:])[0,1])
            if not math.isfinite(rho) or rho>=.95:raise InputError('Nonstable formation residual','missing_data')
        if abs(betas[0]-betas[1])>.5*abs(beta):raise InputError('Unstable split formation beta','missing_data')
        z=float((y[-1]-alpha-beta*x[-1]-residual.mean())/sd);out[50]=result()
        if abs(z)>=2:
            emit(50,'neutral',float(np.exp(alpha+beta*x[-1])),float(rs[-1]['stock']),dict(pair=ds['pair'],pair_direction='short_stock_long_partner' if z>0 else 'long_stock_short_partner',z_score=z,hedge_beta=float(beta),formation_end=rs[59]['_t'].isoformat(),trading_start=rs[60]['_t'].isoformat(),formation_bars=60,trading_bars=5,cost_bps=cost,borrow_assumption=ds['borrow_assumption'],executable_returns=False),rs,ds,'context')
    run((50,),pairs)
    return out


def clean_trades(records,provider=None):
    out=[]
    corrected={r.get('original_trade_id',r.get('trade_id')) for r in records if (r.get('correction') or r.get('cancelled'))}
    corrected.discard(None)
    for r in records:
        conditions=r.get('conditions',[])
        regular=not conditions or (provider=='Alpaca' and isinstance(conditions,list) and conditions==['@'])
        if r.get('trade_id') in corrected or not regular or r.get('correction') or r.get('cancelled'):continue
        out.append(dict(r,price=number(r.get('price'),True),size=number(r.get('size'),True)))
    return out
