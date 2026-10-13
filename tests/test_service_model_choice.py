"""An explicit calculation route must not replace the document model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.service.server import Service, main


class ServiceModelChoiceTests(unittest.TestCase):
    def test_database_cannot_reuse_state_after_adapter_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / 'corpus.json'
            corpus.write_text('{}')
            adapter = root / 'adapter'
            adapter.mkdir()
            config = adapter / 'adapter_config.json'
            weights = adapter / 'adapter_model.safetensors'
            config.write_text('{"r": 8}')
            weights.write_bytes(b'original-weight-fixture')
            database = str(root / 'tasks.sqlite')
            Service(database, str(corpus), 's1', str(adapter))
            Service(database, str(corpus), 's1', str(adapter))
            weights.write_bytes(b'changed-weight-fixture')
            with self.assertRaisesRegex(ValueError, 'model or workflow'):
                Service(database, str(corpus), 's1', str(adapter))
            weights.write_bytes(b'original-weight-fixture')
            config.write_text('{"r": 16}')
            with self.assertRaisesRegex(ValueError, 'model or workflow'):
                Service(database, str(corpus), 's1', str(adapter))

    def test_model_selected_before_generation_from_requested_workflow(self):
        cases = [({}, [], 'reference'),
                 ({'calculator_adapter': 'calculator'}, [], 'calculator'),
                 ({'calculator_adapter': 'calculator'}, ['--reports'], 'reference'),
                 ({'calculator_adapter': 'calculator'},
                  ['--reports', '--neural-index', '/tmp/test-index'], 'reference')]
        for extra_choice, arguments, expected in cases:
            with self.subTest(arguments=arguments, extra_choice=extra_choice):
                choice = {'adapter': 'reference', **extra_choice}
                with patch('sys.argv', ['server', *arguments]), \
                     patch('src.service.server.Path.read_text', return_value=json.dumps(choice)), \
                     patch('src.service.server.logging.basicConfig'), \
                     patch('src.service.server.Service') as service, \
                     patch('src.service.server.ThreadingHTTPServer') as server:
                    server.return_value.serve_forever.side_effect = KeyboardInterrupt
                    main()
                    self.assertEqual(service.call_args.args[3], expected)
                    service.return_value.start.assert_called_once()
                    service.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
