"""Provider safety checks and NumPy CTC decoding regression tests."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app import AssameseTextRecognizer, LiveTextDetectionApp
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


class QuietOcrTests(unittest.TestCase):
    def test_processing_does_not_print_frame_counts_or_periodic_status(self):
        app = LiveTextDetectionApp.__new__(LiveTextDetectionApp)
        app.frame_counter = 0
        app.log_interval = 1
        app.detector = Mock()
        app.recognizer = Mock()
        app.recognizer.recognize.return_value = 'ক'
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        for boxes in (np.array([]), np.array([[1, 1, 10, 10]])):
            with self.subTest(detections=len(boxes)):
                app.detector.detect_one.return_value = (boxes, np.ones(len(boxes)))
                with patch('builtins.print') as output:
                    app.process_frame(frame)
                    output.assert_not_called()


if __name__ == '__main__':
    unittest.main()
