"""Single-host transactional task state, idempotent requests and report versions."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def encoded(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


class Conflict(ValueError):
    pass


class Cancelled(RuntimeError):
    pass


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    payload TEXT NOT NULL, execution TEXT NOT NULL,
                    business TEXT NOT NULL, state TEXT NOT NULL,
                    cancel INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, digest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reports (
                    task_id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(task_id, revision));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, kind TEXT NOT NULL,
                    body TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, body TEXT NOT NULL, expires REAL NOT NULL);
            ''')

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, task_id: str, revision: int, kind: str, body: dict) -> None:
        db.execute('INSERT INTO events(task_id,revision,kind,body,created) VALUES(?,?,?,?,?)',
                   (task_id,revision,kind,encoded(body),time.time()))

    def get(self, task_id: str) -> dict:
        with self.transaction() as db:
            row = db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row is None: raise KeyError(task_id)
            result = dict(row)
            for key in ('payload','state'): result[key] = json.loads(result[key])
            report=db.execute('SELECT body FROM reports WHERE task_id=? AND revision=?',
                              (task_id,row['revision'])).fetchone()
            result['report'] = json.loads(report['body']) if report else None
            return result

    def submit(self, request_id: str, payload: dict, task_id: str | None = None) -> dict:
        """A new request starts or advances a task; the input hash binds retries."""
        if (not isinstance(payload.get('question'), str) or not payload['question'].strip()
                or not isinstance(payload.get('snapshot_id'), str)
                or not payload['snapshot_id']):
            raise ValueError('question and snapshot_id are required')
        if set(payload) - {'question','snapshot_id','company','period','as_of'}:
            raise ValueError('unknown task input fields')
        body = encoded({'task_id':task_id,'payload':payload})
        digest = hashlib.sha256(body.encode()).hexdigest()
        with self.transaction() as db:
            old=db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
            if old:
                if old['digest'] != digest: raise Conflict('request ID input conflict')
                return {'task_id':old['task_id'],'revision':old['revision']}
            queued=db.execute("SELECT count(*) FROM tasks WHERE execution IN ('queued','running')").fetchone()[0]
            if queued >= 64: raise Conflict('queue full')
            revision, history, original_question = 1, [], None
            if task_id:
                old=db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
                if not old: raise KeyError(task_id)
                if old['execution'] in ('running','queued'): raise Conflict('task still active')
                prior=json.loads(old['payload'])
                if old['business'] == 'awaiting_input':
                    original_question = json.loads(old['state']).get('original_question', prior['question'])
                revision=old['revision']+1
                scope=('company','period','as_of','snapshot_id')
                if all(prior.get(k)==payload.get(k) for k in scope):
                    history=json.loads(old['state']).get('history',[])
                    report=db.execute('SELECT body FROM reports WHERE task_id=? AND revision=?',
                                      (task_id,old['revision'])).fetchone()
                    history=history+[{'question':prior['question'],
                                      'report':json.loads(report['body']) if report else None}]
                db.execute('UPDATE tasks SET revision=?,payload=?,execution=?,business=?,state=?,cancel=0 WHERE id=?',
                    (revision,encoded(payload),'queued','blocked',encoded({'history':history, **({'original_question':original_question}
                        if original_question else {})}),task_id))
            else:
                task_id=request_id
                db.execute('INSERT INTO tasks(id,revision,payload,execution,business,state) VALUES(?,?,?,?,?,?)',
                           (task_id,revision,encoded(payload),'queued','blocked',encoded({'history':[]})))
            db.execute('INSERT INTO requests VALUES(?,?,?,?)',(request_id,task_id,revision,digest))
            self.event(db,task_id,revision,'queued',{'scope_changed':not bool(history)})
            return {'task_id':task_id,'revision':revision}

    def claim(self, task_id: str) -> bool:
        with self.transaction() as db:
            row=db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
            if not row: raise KeyError(task_id)
            if row['execution']!='queued' or row['cancel']: return False
            db.execute("UPDATE tasks SET execution='running' WHERE id=?",(task_id,))
            self.event(db,task_id,row['revision'],'running',{})
            return True

    def save(self, task_id: str, revision: int, state: dict, kind: str, body: dict,
             allow_cancelled: bool = False) -> None:
        with self.transaction() as db:
            row=db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row['revision']!=revision or row['execution']!='running': raise Conflict('stale task write')
            if row['cancel'] and not allow_cancelled: raise Cancelled('cancellation requested')
            db.execute('UPDATE tasks SET state=? WHERE id=?',(encoded(state),task_id))
            self.event(db,task_id,revision,kind,body)

    def finish(self, task_id: str, revision: int, business: str, body: dict) -> dict:
        """Commit report, terminal state and event in one transaction."""
        if business not in ('answered','insufficient_evidence'): raise ValueError('invalid report state')
        with self.transaction() as db:
            previous=db.execute('SELECT body FROM reports WHERE task_id=? AND revision=?',(task_id,revision)).fetchone()
            if previous: return json.loads(previous['body'])
            row=db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row['revision']!=revision or row['execution']!='running': raise Conflict('stale report')
            if row['cancel']: raise Cancelled('cancelled before report commit')
            report={**body,'task_id':task_id,'revision':revision,'business':business}
            db.execute('INSERT INTO reports VALUES(?,?,?)',(task_id,revision,encoded(report)))
            db.execute("UPDATE tasks SET execution='succeeded',business=? WHERE id=?",(business,task_id))
            self.event(db,task_id,revision,'report_committed',report)
            return report

    def stop(self, task_id: str, revision: int, execution: str, business: str, detail: dict) -> None:
        if (execution,business) not in {('succeeded','awaiting_input'),('failed','blocked'),
                                     ('cancelled','blocked'),('interrupted','blocked')}:
            raise ValueError('invalid stop state')
        with self.transaction() as db:
            row=db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row['revision']!=revision: raise Conflict('stale stop')
            if row['execution']!='running': return
            if row['cancel']: execution,business='cancelled','blocked'
            state=json.loads(row['state']); state['stop_detail']=detail
            db.execute('UPDATE tasks SET execution=?,business=?,state=? WHERE id=?',
                       (execution,business,encoded(state),task_id))
            self.event(db,task_id,revision,execution,{'business':business,**detail})

    def cancel(self, task_id: str) -> None:
        with self.transaction() as db:
            row=db.execute('SELECT * FROM tasks WHERE id=?',(task_id,)).fetchone()
            if not row: raise KeyError(task_id)
            if row['execution'] not in ('queued','running'): return
            execution='cancelled' if row['execution']=='queued' else 'running'
            db.execute('UPDATE tasks SET cancel=1,execution=? WHERE id=?',(execution,task_id))
            self.event(db,task_id,row['revision'],'cancel_requested',{})

    def recover(self) -> int:
        """Call only after acquiring the exclusive worker lock; no live worker exists."""
        with self.transaction() as db:
            rows=db.execute("SELECT * FROM tasks WHERE execution='running'").fetchall()
            for row in rows:
                state='cancelled' if row['cancel'] else 'queued'
                db.execute('UPDATE tasks SET execution=? WHERE id=?',(state,row['id']))
                self.event(db,row['id'],row['revision'],'recovered',{'execution':state})
            return len(rows)

    def events(self, task_id: str, after: int = 0) -> list[dict]:
        with self.transaction() as db:
            rows=db.execute('SELECT * FROM events WHERE task_id=? AND seq>? ORDER BY seq',
                            (task_id,after)).fetchall()
            return [{**dict(r),'body':json.loads(r['body'])} for r in rows]

    def cached(self, key: dict, value: dict | None = None, ttl: float = 3600) -> dict | None:
        digest=hashlib.sha256(encoded(key).encode()).hexdigest()
        with self.transaction() as db:
            if value is not None:
                db.execute('INSERT OR REPLACE INTO cache VALUES(?,?,?)',(digest,encoded(value),time.time()+ttl))
                return value
            row=db.execute('SELECT * FROM cache WHERE key=? AND expires>?',(digest,time.time())).fetchone()
            return json.loads(row['body']) if row else None
