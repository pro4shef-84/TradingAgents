import pytest
from tradingagents.pilot import market_inputs as m

START='2026-11-06T14:30:00Z'
END='2026-11-06T14:33:00Z'

def bar(t=START):
    return dict(t=t,o=10,h=12,l=9,c=11,v=100,vw=10.5)

def test_paginated_bars_deduplicate_and_report_missing_minutes(monkeypatch):
    pages=iter([{'bars':{'SPY':[bar()]},'next_page_token':'next'},
                {'bars':{'SPY':[bar(),bar('2026-11-06T14:32:00Z')]}}])
    monkeypatch.setattr(m,'alpaca_get',lambda *a,**k:next(pages))
    out=m.fetch_intraday(['SPY'],START,END)
    assert len(out['records']['SPY'])==2
    assert out['coverage']['SPY']['missing_intervals']==['2026-11-06T14:31:00+00:00']
    assert out['coverage']['SPY']['status']=='partial'
    assert out['records']['SPY'][0]['available_at']=='2026-11-06T14:31:00+00:00'
    assert out['feed']=='sip' and out['adjustment']=='raw'

@pytest.mark.parametrize('timestamp',['2026-11-06T14:30:00','garbage',None])
def test_bad_timestamp_rejected(monkeypatch,timestamp):
    monkeypatch.setattr(m,'alpaca_get',lambda *a,**k:{'bars':{'SPY':[bar(timestamp)]}})
    with pytest.raises(ValueError,match='timestamp'):m.fetch_intraday(['SPY'],START,END)

def test_repeated_token_and_page_budget(monkeypatch):
    monkeypatch.setattr(m,'alpaca_get',lambda *a,**k:{'bars':{},'next_page_token':'again'})
    with pytest.raises(ValueError,match='Repeated'):m.fetch_intraday(['SPY'],START,END)
    out=m.fetch_intraday(['SPY'],START,END,max_pages=1)
    assert out['coverage']['SPY']['truncated']

@pytest.mark.parametrize('kind', ['trades','quotes'])
def test_tick_records_keep_conditions_and_raw_fields(monkeypatch,kind):
    row=dict(t=START,p=10,s=5,c=['@'],i=11) if kind=='trades' else dict(t=START,bp=9,ap=10,bs=5,as_=6)
    if kind=='quotes':row['as']=row.pop('as_')
    monkeypatch.setattr(m,'alpaca_get',lambda *a,**k:{kind:{'SPY':[row]}})
    out=getattr(m,'fetch_'+kind)(['SPY'],START,END)
    record=out['records']['SPY'][0]
    assert record['event_at']==record['available_at']
    assert record['provider']=='Alpaca'
    assert out['coverage']['SPY']['status']=='observed'
    assert not out['coverage']['SPY']['complete']

@pytest.mark.parametrize('args', [([],START,END),(['SPY']*21,START,END),(['SPY'],'2026-11-06',END),(['SPY'],END,START)])
def test_bounds_validate_before_network(args):
    with pytest.raises(ValueError):m.fetch_intraday(*args)

def event(**extra):
    return dict(symbol='SPY',event_at=START,available_at=START,source_url='https://example.com/event',provider='reviewed',verified=True,**extra)

def test_sourced_import_availability_cutoff_and_verification():
    result=m.import_records([event()],kind='events',cutoff=END)
    assert result['records']['SPY'][0]['verified']
    with pytest.raises(ValueError):m.import_records([event()|{'source_url':'file:///secret'}],kind='events')
    result=m.import_records([event()|{'available_at':'2026-11-07T15:00:00Z'}],kind='events',cutoff=END)
    assert not result['records']

def test_book_sequence_gap_invalidates_until_snapshot():
    base=event()|{'bids':[[10,5]],'asks':[[11,6]],'sequence':1,'snapshot':True}
    rows=[base,base|{'sequence':3,'snapshot':False},base|{'sequence':4,'snapshot':False},base|{'sequence':7}]
    result=m.import_records(rows,kind='books')
    assert [x['valid'] for x in result['records']['SPY']]==[True,False,False,True]

def test_connections_do_not_claim_configured_is_verified(monkeypatch):
    monkeypatch.setattr(m,'alpaca_configured',lambda:True)
    status=m.connection_status()
    assert status['alpaca']['status']=='configured'
    assert status['x']['status']=='not_connected'
    assert status['level2']['status']=='not_connected'

from tradingagents.pilot import providers

def test_historical_ticks_allowlisted_and_entitlement_typed(monkeypatch):
    class Response:
        status_code=403
    monkeypatch.setattr(providers,'alpaca_headers',lambda:{})
    monkeypatch.setattr(providers.requests,'get',lambda *a,**k:Response())
    with pytest.raises(providers.AlpacaEntitlementError):
        providers.alpaca_get('/v2/stocks/trades',{'feed':'sip'})

def test_transient_network_failure_retried_and_sanitized(monkeypatch):
    calls=[]
    class Response:
        status_code=200
        def json(self):return {'quotes':{}}
    def get(*a,**k):
        calls.append(1)
        if len(calls)<3:raise providers.requests.ConnectionError('secret must never be emitted')
        return Response()
    monkeypatch.setattr(providers,'alpaca_headers',lambda:{})
    monkeypatch.setattr(providers.requests,'get',get)
    monkeypatch.setattr('time.sleep',lambda _:None)
    assert providers.alpaca_get('/v2/stocks/quotes')=={'quotes':{}}
    assert len(calls)==3

def test_retry_budget_and_fixed_endpoint(monkeypatch):
    class Response:status_code=429
    calls=[]
    monkeypatch.setattr(providers,'alpaca_headers',lambda:{})
    monkeypatch.setattr(providers.requests,'get',lambda *a,**k:calls.append(1) or Response())
    monkeypatch.setattr('time.sleep',lambda _:None)
    with pytest.raises(ValueError,match='429'):providers.alpaca_get('/v2/stocks/bars')
    assert len(calls)==3
    with pytest.raises(ValueError,match='Unsupported'):providers.alpaca_get('/v2/orders')

def test_source_projection_and_connection_staleness():
    result=m.import_records([event()],kind='events',cutoff=END)
    assert m.for_symbol(result,'SPY')['records'][0]['event_time']==m.timestamp(START).isoformat()
    status=m.connection_status({'alpaca':{'status':'connected','checked_at':START}},now='2026-11-07T15:00:00Z')
    assert status['alpaca']['status']=='stale'

def test_csv_verified_boolean_is_explicit(tmp_path):
    path=tmp_path/'events.csv'
    path.write_text('symbol,event_at,available_at,source_url,provider,verified\nSPY,'+START+','+START+',https://example.com/event,reviewed,true\n')
    assert m.import_records(path,'events',cutoff=END)['records']['SPY'][0]['verified'] is True

def test_conflicting_duplicate_bar_is_rejected(monkeypatch):
    monkeypatch.setattr(m,'alpaca_get',lambda *a,**k:{'bars':{'SPY':[bar(),bar()|{'c':10}]}})
    with pytest.raises(ValueError,match='Conflicting'):m.fetch_intraday(['SPY'],START,END)

def test_book_prior_event_with_later_available_is_excluded():
    result=m.import_records([event()|{'updated_at':'2026-11-07T15:00:00Z'}],'events',cutoff=END)
    assert not result['records']

def test_unchecked_observation_cannot_claim_connected():
    assert m.connection_status({'alpaca':{'status':'connected'}})['alpaca']['status']!='connected'

def test_daily_bar_pagination_has_hard_budget(monkeypatch):
    calls=[]
    def get(*a,**k):
        calls.append(1)
        return {'bars':{},'next_page_token':str(len(calls))}
    monkeypatch.setattr(providers,'alpaca_get',get)
    with pytest.raises(ValueError,match='budget'):providers.alpaca_daily_bars(['SPY'],'2026-11-06',max_pages=2)
    assert len(calls)==2

def test_connection_scope_is_safe_and_future_checks_are_not_verified(monkeypatch):
    monkeypatch.setattr(m,'alpaca_configured',lambda:True)
    observation={'status':'connected','checked_at':START,'scope':'historical minute only','feed':'sip',
        'capabilities':{'bars':{'status':'connected','record_count':1,'truncated':False,'secret':'hidden'}},
        'budget':{'page_limit':100,'secret':'hidden'},'secret':'hidden'}
    result=m.connection_status({'alpaca':observation},now=END)['alpaca']
    assert result['scope']=='historical minute only'
    assert result['capabilities']['bars']=={'status':'connected','record_count':1,'truncated':False}
    assert result['budget']=={'page_limit':100}
    assert 'hidden' not in str(result)
    result=m.connection_status({'alpaca':observation|{'checked_at':'2099-01-01T00:00:00Z'}},now=END)['alpaca']
    assert result['status']!='connected'


def test_import_projection_declares_adjustment():
    assert m.for_symbol(m.import_records([event()],'events',cutoff=END),'SPY')['adjustment']=='raw'

def test_raw_previous_close_uses_previous_xnys_session(monkeypatch):
    calls=[]
    def get(path,params):
        calls.append((path,dict(params)))
        return {'bars':{'SPY':[dict(t='2026-11-05T05:00:00Z',c=550)]}}
    monkeypatch.setattr(m,'alpaca_get',get)
    result=m.fetch_previous_closes(['SPY'],'2026-11-06')
    assert result['SPY']['price']==550 and result['SPY']['asof']=='2026-11-05'
    assert result['SPY']['adjustment']=='raw' and result['SPY']['feed']=='sip'
    assert calls[0][1]['timeframe']=='1Day' and calls[0][1]['adjustment']=='raw'


def test_previous_close_rejects_stale_and_budget_exhaustion(monkeypatch):
    monkeypatch.setattr(m,'alpaca_get',lambda *a,**k:{'bars':{'SPY':[dict(t='2026-11-04T05:00:00Z',c=550)]}})
    with pytest.raises(ValueError,match='session'):m.fetch_previous_closes(['SPY'],'2026-11-06')
    monkeypatch.setattr(m,'alpaca_get',lambda *a,**k:{'bars':{},'next_page_token':'next'})
    with pytest.raises(ValueError,match='budget'):m.fetch_previous_closes(['SPY'],'2026-11-06',max_pages=1)
