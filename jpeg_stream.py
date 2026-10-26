"""Receive the host jpegenc/tcpserversink stream without loading GStreamer."""
import socket
import threading
from urllib.parse import urlsplit

import cv2
import numpy as np


def extract_frames(buffer):
    """Remove complete jpegenc frames, retaining a partial frame between reads."""
    while True:
        start = buffer.find(b'\xff\xd8')
        if start < 0:
            # A JPEG start marker can be split across TCP packets.
            buffer[:] = buffer[-1:]
            return
        if start:
            del buffer[:start]
        end = buffer.find(b'\xff\xd9', 2)
        if end < 0:
            if len(buffer) > 16 * 1024 * 1024:
                raise RuntimeError('Host JPEG frame exceeds the 16 MB capture limit.')
            return
        frame = bytes(buffer[:end + 2])
        del buffer[:end + 2]
        yield frame


class TcpJpegCapture:
    """Keep the newest compressed frame so slow inference cannot build a backlog."""
    def __init__(self, source):
        address = urlsplit(source)
        if (address.scheme != 'tcpjpeg' or not address.hostname or address.path not in ('', '/')
                or address.query or address.fragment or address.username or address.password):
            raise ValueError('Use tcpjpeg://HOST:PORT for the host JPEG stream.')
        try:
            self.socket = socket.create_connection((address.hostname, address.port or 8554), timeout=10)
        except OSError as exc:
            raise RuntimeError('Cannot connect to the host camera sender. Start '
                               'bash scripts/stream_csi_camera.sh on the Jetson host first.') from exc
        self.condition = threading.Condition()
        self.closed = False
        self.error = None
        self.latest = None
        self.sequence = 0
        self.consumed = 0
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        buffer = bytearray()
        try:
            while not self.closed:
                data = self.socket.recv(65536)
                if not data:
                    raise RuntimeError('Host camera sender disconnected. Restart the sender and OCR service.')
                buffer.extend(data)
                for jpeg in extract_frames(buffer):
                    with self.condition:
                        self.latest = jpeg
                        self.sequence += 1
                        self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                if not self.closed:
                    self.error = str(exc)
                self.condition.notify_all()

    def read(self):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence != self.consumed or self.error or self.closed, timeout=10)
            if self.closed:
                return False, None
            if not ready:
                raise RuntimeError('No JPEG frame received from the host camera sender within 10 seconds.')
            if self.sequence == self.consumed:
                raise RuntimeError(f'Host camera stream failed: {self.error}')
            jpeg = self.latest
            self.consumed = self.sequence
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError('Host camera stream delivered an invalid JPEG frame.')
        return True, frame

    def release(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        self.thread.join(timeout=1)
