"""JPEG framing, decoding, and failure handling without Jetson camera plugins."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from jpeg_stream import TcpJpegCapture, extract_frames


class JpegStreamTests(unittest.TestCase):
    def test_fragmented_markers_and_multiple_frames(self):
        buffer = bytearray(b'partial old frame\xff')
        self.assertEqual(list(extract_frames(buffer)), [])
        buffer.extend(b'\xd8one\xff')
        self.assertEqual(list(extract_frames(buffer)), [])
        buffer.extend(b'\xd9\xff\xd8two\xff\xd9\xff\xd8partial')
        self.assertEqual(list(extract_frames(buffer)), [b'\xff\xd8one\xff\xd9', b'\xff\xd8two\xff\xd9'])
        self.assertEqual(buffer, b'\xff\xd8partial')

    def test_incomplete_frame_has_bounded_memory(self):
        with self.assertRaisesRegex(RuntimeError, '16 MB'):
            list(extract_frames(bytearray(b'\xff\xd8' + b'x' * (16 * 1024 * 1024))))

    @patch('jpeg_stream.socket.create_connection')
    def test_latest_frame_is_decoded_and_disconnect_is_reported(self, connect):
        dark = cv2.imencode('.jpg', np.zeros((32, 64, 3), dtype=np.uint8))[1].tobytes()
        light = cv2.imencode('.jpg', np.full((32, 64, 3), 240, dtype=np.uint8))[1].tobytes()
        connect.return_value.recv.side_effect = [dark[:5], dark[5:] + light, b'']
        capture = TcpJpegCapture('tcpjpeg://127.0.0.1:8554')
        try:
            capture.thread.join(timeout=2)
            self.assertFalse(capture.thread.is_alive())
            ok, frame = capture.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (32, 64, 3))
            self.assertGreater(frame.mean(), 230)
            with self.assertRaisesRegex(RuntimeError, 'sender disconnected'):
                capture.read()
        finally:
            capture.release()
        connect.assert_called_once_with(('127.0.0.1', 8554), timeout=10)
        connect.return_value.close.assert_called_once()

    @patch('jpeg_stream.socket.create_connection', side_effect=ConnectionRefusedError())
    def test_missing_sender_has_actionable_error(self, connect):
        with self.assertRaisesRegex(RuntimeError, 'stream_csi_camera.sh'):
            TcpJpegCapture('tcpjpeg://127.0.0.1:8554')

    @patch('jpeg_stream.socket.create_connection')
    def test_invalid_url_does_not_connect(self, connect):
        with self.assertRaises(ValueError):
            TcpJpegCapture('tcpjpeg://host/path')
        connect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
