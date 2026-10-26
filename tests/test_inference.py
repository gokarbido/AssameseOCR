"""Provider safety checks and NumPy CTC decoding regression tests."""
import unittest
from unittest.mock import patch

import numpy as np

from app import AssameseTextRecognizer
from inference_runtime import create_session


class ProviderTests(unittest.TestCase):
    @patch('inference_runtime.ort.InferenceSession')
    @patch('inference_runtime.ort.get_available_providers', return_value=['CPUExecutionProvider'])
    def test_cuda_missing_fails_before_loading(self, available, constructor):
        with self.assertRaisesRegex(RuntimeError, 'Missing ONNX Runtime providers'):
            create_session('model.onnx')
        constructor.assert_not_called()

    @patch('inference_runtime.ort.InferenceSession')
    @patch('inference_runtime.ort.get_available_providers', return_value=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    def test_startup_cpu_fallback_is_rejected(self, available, constructor):
        constructor.return_value.get_providers.return_value = ['CPUExecutionProvider']
        with self.assertRaisesRegex(RuntimeError, 'refusing CPU-only fallback'):
            create_session('model.onnx')

    @patch('inference_runtime.ort.InferenceSession')
    @patch('inference_runtime.ort.get_available_providers', return_value=['CPUExecutionProvider'])
    def test_explicit_cpu_works(self, available, constructor):
        constructor.return_value.get_providers.return_value = ['CPUExecutionProvider']
        session = create_session('model.onnx', ['CPUExecutionProvider'])
        session.disable_fallback.assert_called_once()

    @patch('inference_runtime.ort.InferenceSession')
    @patch('inference_runtime.ort.get_available_providers', return_value=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    def test_cuda_session_disables_runtime_fallback(self, available, constructor):
        constructor.return_value.get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = create_session('model.onnx')
        session.disable_fallback.assert_called_once()


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = AssameseTextRecognizer.__new__(AssameseTextRecognizer)
        self.recognizer.blank = '<blank>'
        self.recognizer.vocab = ['<blank>', 'ক', 'ৰ']
        self.recognizer.idx_to_char = dict(enumerate(self.recognizer.vocab))

    def test_blanks_separate_repeated_characters(self):
        logits = np.full((1, 7, 3), -1000.0, dtype=np.float32)
        logits[0, np.arange(7), [0, 1, 1, 0, 1, 2, 2]] = 1000
        self.assertEqual(self.recognizer.ctc_decode(logits), 'ককৰ')

    def test_all_blank(self):
        logits = np.zeros((1, 4, 3), dtype=np.float32)
        self.assertEqual(self.recognizer.ctc_decode(logits), '')


if __name__ == '__main__':
    unittest.main()
