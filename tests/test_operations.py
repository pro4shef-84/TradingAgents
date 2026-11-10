import json
import tempfile
import unittest
from pathlib import Path


class OperationTests(unittest.TestCase):
    def test_imports_are_saved_without_implying_live_connection(self):
        from tradingagents.pilot.operations import import_data
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'events.json'
            source.write_text(json.dumps([{'symbol':'A','provider':'reviewed company release',
                'source_url':'https://example.com/release','event_time':'2026-11-06T15:00:00Z',
                'available_at':'2026-11-06T15:01:00Z','verified':True,'direction':'bullish'}]))
            result=import_data(Path(tmp),source,'events')
            self.assertEqual(result['status'],'imported')
            data=json.loads(Path(result['path']).read_text())
            self.assertTrue(data['records']['A'][0]['verified'])
            self.assertFalse(data['coverage']['A']['complete'])
