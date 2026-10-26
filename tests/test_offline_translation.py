"""Offline setup errors should be actionable without fetching remote artifacts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline_translation import OfflineTranslator


class OfflineSetupTests(unittest.TestCase):
    def test_missing_weights_fail_without_a_network_request(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('socket.create_connection', side_effect=AssertionError('No network allowed')):
                with self.assertRaisesRegex(RuntimeError, 'download_translation_model.py'):
                    OfflineTranslator(Path(directory))


if __name__ == '__main__':
    unittest.main()
