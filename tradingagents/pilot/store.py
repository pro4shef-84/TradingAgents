"""Durable, append-only snapshots and observations for the local pilot."""
import json
import sqlite3
from pathlib import Path
from tradingagents.pilot.screen import utcnow

class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY,at TEXT,ticker TEXT,payload TEXT);
            CREATE INDEX IF NOT EXISTS ticker_snapshot ON snapshots(ticker,id);
            CREATE TABLE IF NOT EXISTS quotes(id INTEGER PRIMARY KEY,at TEXT,ticker TEXT,payload TEXT);
            CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY,at TEXT,ticker TEXT,snapshot_id INTEGER,stance TEXT,note TEXT);
            ''')
    def connect(self):
        c=sqlite3.connect(self.path,timeout=20);c.row_factory=sqlite3.Row
        return c
    def save(self,row):
        with self.connect() as c:
            return c.execute('INSERT INTO snapshots(at,ticker,payload) VALUES(?,?,?)',(utcnow(),row['ticker'],json.dumps(row,allow_nan=False))).lastrowid
    def latest(self):
        with self.connect() as c:
            return [dict(json.loads(r['payload']),snapshot_id=r['id']) for r in c.execute('SELECT s.* FROM snapshots s JOIN (SELECT ticker,max(id) id FROM snapshots GROUP BY ticker) x ON s.id=x.id')]
    def quote(self,ticker,q):
        with self.connect() as c:c.execute('INSERT INTO quotes(at,ticker,payload) VALUES(?,?,?)',(utcnow(),ticker,json.dumps(q,allow_nan=False)))
    def samples(self,ticker,provider):
        with self.connect() as c:
            rows=[json.loads(r['payload']) for r in c.execute('SELECT payload FROM quotes WHERE ticker=? ORDER BY id DESC LIMIT 100',(ticker,))]
        return [q for q in rows if q.get('provider')==provider]
