"""Exercise shared streaming and readiness without a camera or GPU."""
import time
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

from web_app import LiveFeed, create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.feed = LiveFeed(device='cpu')
        self.client = create_app(self.feed).test_client()

    def publish(self):
        with self.feed.condition:
            self.feed.jpeg = b'jpeg-frame'
            self.feed.sequence += 1
            self.feed.updated = time.monotonic()
            self.feed.texts = ['অসমীয়া']
            self.feed.fps = 12.5
            self.feed.condition.notify_all()

    def test_page_and_initial_readiness(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Recognized text', response.data)
        self.assertEqual(self.client.get('/healthz').status_code, 503)
        self.assertEqual(self.client.get('/api/status').json['state'], 'waiting')

    def test_health_requires_recent_frames_and_no_error(self):
        self.publish()
        self.assertEqual(self.client.get('/healthz').status_code, 200)
        self.assertEqual(self.client.get('/api/status').json['texts'], ['অসমীয়া'])
        self.feed.updated -= 16
        self.assertEqual(self.client.get('/healthz').status_code, 503)
        self.assertEqual(self.client.get('/api/status').json['texts'], [])
        self.feed.error = 'Camera disconnected'
        self.assertEqual(self.client.get('/api/status').json['state'], 'error')

    def test_viewers_share_frame_without_consuming_it(self):
        self.publish()
        first, second = self.feed.frames(), self.feed.frames()
        self.assertEqual(next(first), next(second))
        self.publish()
        self.assertIn(b'Content-Length: 10\r\n', next(first))
        first.close()
        self.assertIn(b'jpeg-frame', next(second))
        self.feed.stop()
        with self.assertRaises(StopIteration):
            next(second)

    def test_stream_response_has_multipart_and_no_cache(self):
        self.publish()
        response = self.client.get('/video', buffered=False)
        self.assertIn('boundary=frame', response.content_type)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertIn(b'Content-Type: image/jpeg', next(response.response))
        response.close()

    @patch('web_app.LiveTextDetectionApp')
    @patch('web_app.cv2.VideoCapture')
    def test_worker_encodes_inference_and_releases_camera(self, capture, pipeline):
        camera = capture.return_value
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, np.zeros((32, 64, 3), dtype=np.uint8)), (False, None)]
        pipeline.return_value.process_frame.return_value = ([], [], ['অসমীয়া'])
        pipeline.return_value.draw_detections.side_effect = lambda frame, *args: frame
        with self.assertLogs(level='ERROR'):
            self.feed.start()
            self.feed.thread.join(timeout=5)
        self.assertFalse(self.feed.thread.is_alive())
        self.assertEqual(self.feed.sequence, 1)
        self.assertTrue(self.feed.jpeg.startswith(b'\xff\xd8'))
        self.assertIn('stopped delivering', self.feed.error)
        camera.release.assert_called_once()
        capture.assert_called_once_with(0, cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY)

    @patch('web_app.sys.platform', 'linux')
    @patch('web_app.cv2.VideoCapture')
    def test_linux_usb_bypasses_gstreamer_and_retries_failed_mode(self, capture):
        failed, working = Mock(), Mock()
        failed.isOpened.return_value = working.isOpened.return_value = True
        failed.read.return_value = (False, None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        working.read.return_value = (True, frame)
        capture.side_effect = [failed, working]
        with self.assertLogs(level='WARNING'):
            cap, first = self.feed._open_camera()
        self.assertIs(cap, working)
        self.assertIs(first, frame)
        self.assertEqual(capture.call_args_list[0].args, (0, cv2.CAP_V4L2))
        failed.release.assert_called_once()
        working.release.assert_not_called()
        working.set.assert_any_call(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    @patch('web_app.sys.platform', 'linux')
    @patch('web_app.cv2.VideoCapture')
    def test_unavailable_camera_releases_all_attempts(self, capture):
        capture.return_value.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'Cannot read camera'):
            self.feed._open_camera()
        self.assertEqual(capture.return_value.release.call_count, 3)

    @patch('web_app.sys.platform', 'linux')
    @patch('web_app.cv2.VideoCapture')
    def test_stream_keeps_automatic_backend_and_source_resolution(self, capture):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        capture.return_value.read.return_value = (True, frame)
        LiveFeed(camera='rtsp://camera/live')._open_camera()
        capture.assert_called_once_with('rtsp://camera/live', cv2.CAP_ANY)
        self.assertNotIn(cv2.CAP_PROP_FRAME_WIDTH,
                         [call.args[0] for call in capture.return_value.set.call_args_list])

    @patch('web_app.cv2.getBuildInformation', return_value='Video I/O:\n    GStreamer: YES (1.20)')
    @patch('web_app.cv2.VideoCapture')
    def test_csi_pipeline_uses_gstreamer_without_usb_mode_changes(self, capture, build_info):
        pipeline = ('nvarguscamerasrc sensor-id=0 ! nvvidconv ! video/x-raw,format=BGRx ! '
                    'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false')
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        capture.return_value.read.return_value = (True, frame)
        cap, first = LiveFeed(camera=pipeline, gstreamer=True)._open_camera()
        capture.assert_called_once_with(pipeline, cv2.CAP_GSTREAMER)
        self.assertIs(first, frame)
        properties = [call.args[0] for call in cap.set.call_args_list]
        self.assertNotIn(cv2.CAP_PROP_FOURCC, properties)
        self.assertNotIn(cv2.CAP_PROP_FRAME_WIDTH, properties)

    @patch('web_app.cv2.getBuildInformation', return_value='GStreamer: NO')
    @patch('web_app.cv2.VideoCapture')
    def test_csi_reports_opencv_without_gstreamer(self, capture, build_info):
        with self.assertRaisesRegex(RuntimeError, 'OpenCV has no GStreamer support'):
            LiveFeed(camera='nvarguscamerasrc ! appsink', gstreamer=True)._open_camera()
        capture.assert_not_called()

    @patch('web_app.cv2.getBuildInformation')
    @patch('web_app.cv2.VideoCapture')
    @patch('web_app.TcpJpegCapture')
    def test_host_jpeg_bypasses_all_gstreamer_and_video_capture(self, jpeg_capture, capture, build_info):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        jpeg_capture.return_value.read.return_value = (True, frame)
        cap, first = LiveFeed(camera='tcpjpeg://127.0.0.1:8554')._open_camera()
        self.assertIs(first, frame)
        self.assertIs(cap, jpeg_capture.return_value)
        capture.assert_not_called()
        build_info.assert_not_called()

    @patch('web_app.LiveTextDetectionApp', side_effect=RuntimeError('Missing CUDA'))
    @patch('web_app.cv2.VideoCapture')
    def test_model_failure_is_reported_without_opening_camera(self, capture, pipeline):
        with self.assertLogs(level='ERROR'):
            self.feed._capture()
        capture.assert_not_called()
        self.assertEqual(self.client.get('/api/status').json['error'], 'Missing CUDA')
        self.assertEqual(self.client.get('/healthz').status_code, 503)


if __name__ == '__main__':
    unittest.main()
