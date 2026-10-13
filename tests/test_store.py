import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from src.harness.store import Cancelled, Conflict, Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'state.sqlite'))
        self.payload={'question':'Revenue?','company':'A','period':'2023','snapshot_id':'s1'}

    def tearDown(self):
        self.tmp.cleanup()

    def start(self):
        self.store.submit('r1',self.payload)
        self.assertTrue(self.store.claim('r1'))

    def test_request_id_is_bound_to_input(self):
        first=self.store.submit('r1',self.payload)
        self.assertEqual(first,self.store.submit('r1',self.payload))
        self.assertEqual(len(self.store.events('r1')),1)
        with self.assertRaises(Conflict):
            self.store.submit('r1',{**self.payload,'company':'B'})

    def test_concurrent_claim_has_one_owner(self):
        self.store.submit('r1',self.payload)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _: self.store.claim('r1'),range(2)))
        self.assertEqual(sorted(results),[False,True])

    def test_report_and_event_rollback_together(self):
        self.start()
        with patch.object(Store,'event',side_effect=RuntimeError('crash before commit')):
            with self.assertRaises(RuntimeError):
                self.store.finish('r1',1,'answered',{'value':12})
        self.assertIsNone(self.store.get('r1')['report'])
        self.assertEqual(self.store.get('r1')['execution'],'running')

    def test_crash_after_commit_does_not_duplicate_report(self):
        self.start()
        report=self.store.finish('r1',1,'answered',{'value':12})
        recovered=Store(self.store.path)
        self.assertEqual(recovered.recover(),0)
        self.assertEqual(recovered.finish('r1',1,'answered',{'value':99}),report)
        self.assertEqual(sum(e['kind']=='report_committed' for e in recovered.events('r1')),1)

    def test_cancel_before_commit_prevents_report(self):
        self.start()
        self.store.cancel('r1')
        self.assertEqual(self.store.get('r1')['execution'],'running')
        with self.assertRaises(Cancelled): self.store.finish('r1',1,'answered',{})
        self.store.stop('r1',1,'failed','blocked',{})
        self.assertEqual(self.store.get('r1')['execution'],'cancelled')
        self.assertIsNone(self.store.get('r1')['report'])

    def test_cancel_queued_and_completed(self):
        self.store.submit('r1',self.payload)
        self.store.cancel('r1')
        self.assertFalse(self.store.claim('r1'))
        self.assertEqual(self.store.get('r1')['execution'],'cancelled')
        self.store.submit('r2',self.payload)
        self.store.claim('r2')
        self.store.finish('r2',1,'answered',{})
        self.store.cancel('r2')
        self.assertEqual(self.store.get('r2')['execution'],'succeeded')

    def test_recovery_preserves_budget_and_observations(self):
        self.start()
        state={'attempts':3,'history':[],'observations':[{'value':7}]}
        self.store.save('r1',1,state,'observation',{})
        self.assertEqual(self.store.recover(),1)
        self.assertEqual(self.store.get('r1')['state'],state)
        self.assertTrue(self.store.claim('r1'))

    def test_clarification_is_not_an_answer(self):
        self.start()
        self.store.stop('r1',1,'succeeded','awaiting_input',{'question':'Which year?'})
        row=self.store.get('r1')
        self.assertEqual(row['business'],'awaiting_input')
        self.assertIsNone(row['report'])
        receipt=self.store.submit('resume1',{**self.payload,'question':'2022','period':'2022'},'r1')
        self.assertEqual(receipt['revision'],2)
        self.assertEqual(self.store.get('r1')['state'],{'history':[], 'original_question':'Revenue?'})
        self.assertEqual(receipt,self.store.submit('resume1',{**self.payload,'question':'2022','period':'2022'},'r1'))

    def test_same_scope_history_but_no_old_facts(self):
        self.start()
        self.store.save('r1',1,{'history':[],'facts':{'stale':1}},'observation',{})
        self.store.finish('r1',1,'answered',{'value':12})
        self.store.submit('resume1',{**self.payload,'question':'And profit?'},'r1')
        state=self.store.get('r1')['state']
        self.assertNotIn('facts',state)
        self.assertEqual(state['history'][0]['report']['value'],12)
        with self.assertRaises(Conflict): self.store.save('r1',1,{},'stale',{})

    def test_cache_scope_and_expiration(self):
        key={'tool':'read','version':'1','input':'e1','snapshot':'s1','company':'A','period':'2023','as_of':None}
        self.store.cached(key,{'value':12})
        self.assertEqual(self.store.cached(key),{'value':12})
        for field in key:
            self.assertIsNone(self.store.cached({**key,field:'different'}))
        self.store.cached(key,{'value':13},ttl=-1)
        self.assertIsNone(self.store.cached(key))

    def test_event_cursor_reconnect(self):
        self.start()
        events=self.store.events('r1')
        self.assertEqual(self.store.events('r1',events[0]['seq']),events[1:])
        self.assertEqual(self.store.events('r1',events[-1]['seq']),[])

    def test_queue_bound_and_bad_payload(self):
        with self.assertRaises(ValueError): self.store.submit('bad',{})
        for i in range(64): self.store.submit(str(i),self.payload)
        with self.assertRaisesRegex(Conflict,'queue full'): self.store.submit('overflow',self.payload)
