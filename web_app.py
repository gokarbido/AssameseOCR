"""One camera/inference worker shared by a small number of browser viewers."""
import argparse
import logging
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template
from waitress import serve

from app import LiveTextDetectionApp
from jpeg_stream import TcpJpegCapture


class LiveFeed:
    def __init__(self, camera='0', device='cuda', gstreamer=False,
                 width=640, height=480, fps=30, fourcc='MJPG'):
        self.camera = camera
        self.device = device
        self.gstreamer = gstreamer
        self.width, self.height, self.capture_fps = width, height, fps
        self.fourcc = fourcc
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.jpeg = None
        self.sequence = 0
        self.updated = 0.0
        self.texts = []
        self.fps = 0.0
        self.error = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=5)

    def _open_camera(self):
        if self.camera.startswith('tcpjpeg://'):
            cap = TcpJpegCapture(self.camera)
            try:
                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    raise RuntimeError('Host camera sender provided no frame.')
                logging.info('Host JPEG camera connected at %s; capture does not use GStreamer', self.camera)
                return cap, first_frame
            except Exception:
                cap.release()
                raise
        if self.gstreamer and not re.search(r'GStreamer:\s+YES', cv2.getBuildInformation()):
            raise RuntimeError('OpenCV has no GStreamer support. Use JetPack system OpenCV (python3-opencv).')
        source = int(self.camera) if self.camera.isdecimal() and not self.gstreamer else self.camera
        usb = (not self.gstreamer and sys.platform.startswith('linux')
               and (isinstance(source, int) or str(source).startswith('/dev/video')))
        backend = cv2.CAP_GSTREAMER if self.gstreamer else cv2.CAP_V4L2 if usb else cv2.CAP_ANY
        # Probe an actual frame, since isOpened() can succeed for an unusable mode.
        modes = [(self.fourcc, self.width, self.height, self.capture_fps),
                 ('YUYV', 640, 480, 30), None] if usb else [None]
        for mode in modes:
            cap = cv2.VideoCapture(source, backend)
            try:
                if not cap.isOpened():
                    cap.release()
                    continue
                if mode:
                    codec, width, height, fps = mode
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, first_frame = cap.read()
                if ok and first_frame is not None:
                    logging.info('Camera %s opened with backend=%s, mode=%s, actual=%sx%s @ %s FPS',
                                 source, backend, mode or 'source defaults',
                                 cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                 cap.get(cv2.CAP_PROP_FPS))
                    return cap, first_frame
                logging.warning('Camera %s produced no frame with mode %s', source, mode)
            except Exception:
                cap.release()
                raise
            # Release a failed attempt before trying another format.
            cap.release()
        raise RuntimeError(
            f'Cannot read camera {source}. Check CAMERA_DEVICE, supported formats '
            '(v4l2-ctl --list-formats-ext), and whether another process owns the camera. '
            'For CSI/GStreamer, the host JetPack and container libraries must match.')

    def _capture(self):
        cap = None
        try:
            root = Path(__file__).resolve().parent
            pipeline = LiveTextDetectionApp(str(root / 'det_asm.onnx'),
                                            str(root / 'model.onnx'),
                                            str(root / 'config.yaml'), device=self.device)
            cap, first_frame = self._open_camera()
            while not self.stopped.is_set():
                started = time.monotonic()
                if first_frame is not None:
                    ok, frame, first_frame = True, first_frame, None
                else:
                    ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError('Camera stopped delivering frames. Check it and restart the service.')
                boxes, scores, texts = pipeline.process_frame(frame)
                annotated = pipeline.draw_detections(frame, boxes, scores, texts)
                ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError('Could not encode the camera frame.')
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.texts = [text for text in texts if text.strip()]
                    self.fps = round(1 / max(time.monotonic() - started, 0.001), 1)
                    self.updated = time.monotonic()
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            logging.exception('Live feed stopped')
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
        finally:
            if cap is not None:
                cap.release()

    def status(self):
        with self.condition:
            live = (self.jpeg is not None and not self.error and not self.stopped.is_set()
                    and time.monotonic() - self.updated < 15)
            state = 'live' if live else ('error' if self.error else 'waiting')
            return dict(state=state, error=self.error, fps=self.fps if live else 0,
                        texts=list(self.texts) if live else [], frames=self.sequence)

    def frames(self):
        previous = -1
        while not self.stopped.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.sequence != previous or self.error or self.stopped.is_set(), timeout=5)
                if self.error or self.stopped.is_set():
                    return
                if self.jpeg is None or previous == self.sequence:
                    previous = self.sequence
                    continue
                previous, jpeg = self.sequence, self.jpeg
            yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                   + str(len(jpeg)).encode('ascii') + b'\r\n\r\n' + jpeg + b'\r\n')


def create_app(feed):
    app = Flask(__name__)

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/video')
    def video():
        return Response(feed.frames(), mimetype='multipart/x-mixed-replace; boundary=frame',
                        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @app.get('/api/status')
    def status():
        response = jsonify(feed.status())
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/healthz')
    def health():
        state = feed.status()
        return jsonify(state=state['state']), 200 if state['state'] == 'live' else 503

    return app


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Assamese OCR browser viewer')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--camera', default=os.environ.get('CAMERA_SOURCE', '0'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--gstreamer', action='store_true',
                        default=os.environ.get('CAMERA_GSTREAMER', '0').lower() in ('1', 'true', 'yes'),
                        help='Use a GStreamer pipeline (also CAMERA_GSTREAMER=1)')
    parser.add_argument('--width', type=int, default=int(os.environ.get('CAMERA_WIDTH', '640')))
    parser.add_argument('--height', type=int, default=int(os.environ.get('CAMERA_HEIGHT', '480')))
    parser.add_argument('--fps', type=int, default=int(os.environ.get('CAMERA_FPS', '30')))
    parser.add_argument('--fourcc', default=os.environ.get('CAMERA_FOURCC', 'MJPG'))
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) < 1 or len(args.fourcc) != 4:
        parser.error('width, height and fps must be positive; fourcc must be four characters')
    feed = LiveFeed(args.camera, args.device, args.gstreamer,
                    args.width, args.height, args.fps, args.fourcc)

    def shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    feed.start()
    try:
        serve(create_app(feed), host=args.host, port=args.port, threads=8)
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()


if __name__ == '__main__':
    main()
