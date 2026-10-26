"""One camera/inference worker shared by a small number of browser viewers."""
import argparse
import logging
import os
from pathlib import Path
import signal
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template
from waitress import serve

from app import LiveTextDetectionApp


class LiveFeed:
    def __init__(self, camera='0', device='cuda', gstreamer=False):
        self.camera = camera
        self.device = device
        self.gstreamer = gstreamer
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

    def _capture(self):
        cap = None
        try:
            root = Path(__file__).resolve().parent
            pipeline = LiveTextDetectionApp(str(root / 'det_asm.onnx'),
                                            str(root / 'model.onnx'),
                                            str(root / 'config.yaml'), device=self.device)
            source = int(self.camera) if self.camera.isdecimal() and not self.gstreamer else self.camera
            cap = (cv2.VideoCapture(source, cv2.CAP_GSTREAMER) if self.gstreamer
                   else cv2.VideoCapture(source))
            if not cap.isOpened():
                raise RuntimeError('Cannot open camera. Check the camera source and device access.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stopped.is_set():
                started = time.monotonic()
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
    parser = argparse.ArgumentParser(description='Assamese OCR browser viewer')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--camera', default=os.environ.get('CAMERA_SOURCE', '0'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--gstreamer', action='store_true')
    args = parser.parse_args()
    feed = LiveFeed(args.camera, args.device, args.gstreamer)

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
