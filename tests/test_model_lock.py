import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.model.agent import ModelDriver


class ModelLockTests(unittest.TestCase):
    def test_lock_applies_to_another_working_directory(self):
        with patch('src.model.agent.AutoTokenizer.from_pretrained'), patch('src.model.agent.load_model'):
            with ModelDriver() as model, tempfile.TemporaryDirectory() as directory:
                self.assertTrue(model.lock.name.startswith('/tmp/'))
                code = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
                        'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
                result = subprocess.run(['python', '-c', code, model.lock.name],
                                        cwd=directory, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('BlockingIOError', result.stderr)


if __name__ == '__main__':
    unittest.main()
