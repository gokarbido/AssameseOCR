"""Timestamped CSI experiment: GStreamer source PTS to application publication.

This excludes browser/network display and does not claim physical exposure time.
Run sender on the Jetson host, consumer in the existing CUDA Docker image.
"""
import argparse
import csv
import json
from pathlib import Path
import socket
import struct
import threading
import time

HEADER = struct.Struct('!QQI')


def receive_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        part = connection.recv(size-len(data))
        if not part:
            raise EOFError('Timestamped camera sender disconnected')
        data.extend(part)
    return bytes(data)


def sender(port):
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen(1)
    connection, _ = server.accept()
    pipeline = Gst.parse_launch('nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=I420 ! jpegenc quality=85 ! appsink name=output emit-signals=true sync=false max-buffers=1 drop=true')
    loop, sequence = GLib.MainLoop(), 0
    def sample(sink):
        nonlocal sequence
        buffer = sink.emit('pull-sample').get_buffer()
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            raise RuntimeError('Camera has no PTS; cannot measure source age.')
        clock = pipeline.get_clock()
        age_ns = clock.get_time() - (pipeline.get_base_time()+buffer.pts)
        timestamp = time.monotonic_ns()-age_ns
        jpeg = buffer.extract_dup(0, buffer.get_size())
        sequence += 1
        try:
            connection.sendall(HEADER.pack(sequence, timestamp, len(jpeg))+jpeg)
        except OSError:
            loop.quit()
            return Gst.FlowReturn.EOS
        return Gst.FlowReturn.OK
    pipeline.get_by_name('output').connect('new-sample', sample)
    try:
        pipeline.set_state(Gst.State.PLAYING)
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        connection.close()
        server.close()


def consumer(args):
    import cv2
    import numpy as np
    from app import LiveTextDetectionApp
    from web_app import LiveFeed
    from benchmark import distribution, write_json
    root = Path(__file__).resolve().parent
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    pipeline = LiveTextDetectionApp(str(root/'det_asm.onnx'),str(root/'model.onnx'),str(root/'config.yaml'),device='cuda')
    feed = LiveFeed()
    connection = socket.create_connection(('127.0.0.1',args.port),timeout=15)
    connection.settimeout(15)
    condition, latest, failure = threading.Condition(), None, None
    def receive():
        nonlocal latest, failure
        try:
            while True:
                sequence, timestamp, size = HEADER.unpack(receive_exact(connection,HEADER.size))
                if not 0 < size <= 16*1024*1024:
                    raise ValueError('Invalid JPEG length')
                jpeg = receive_exact(connection,size)
                with condition:
                    latest = sequence,timestamp,jpeg
                    condition.notify_all()
        except Exception as error:
            with condition:
                failure = str(error)
                condition.notify_all()
    thread = threading.Thread(target=receive,daemon=True)
    thread.start()
    summaries, previous = [], -1
    with (output/'frames.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['repeat','sequence','source_pts_monotonic_ns','publication_monotonic_ns','age_ms','regions'])
        writer.writeheader()
        for repeat in range(1,4):
            start, ages, counts = time.monotonic(), [], []
            while time.monotonic()-start < args.seconds:
                with condition:
                    if not condition.wait_for(lambda: failure or (latest and latest[0] != previous),timeout=15):
                        raise RuntimeError('No timestamped camera frame')
                    if failure:
                        raise RuntimeError(failure)
                    sequence,timestamp,jpeg = latest
                    previous = sequence
                frame = cv2.imdecode(np.frombuffer(jpeg,dtype=np.uint8),cv2.IMREAD_COLOR)
                boxes,scores,texts = pipeline.process_frame(frame)
                annotated = pipeline.draw_detections(frame,boxes,scores,texts,[None]*len(texts))
                ok,encoded = cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,80])
                if not ok:
                    raise RuntimeError('Publication encoding failed')
                with feed.condition:
                    feed.jpeg = encoded.tobytes()
                    feed.sequence += 1
                    feed.updated = time.monotonic()
                    published = time.monotonic_ns()
                age = (published-timestamp)/1e6
                if not 0 <= age < 10000:
                    raise RuntimeError('Invalid clock mapping or camera stall')
                ages.append(age); counts.append(len(boxes))
                writer.writerow(dict(repeat=repeat,sequence=sequence,source_pts_monotonic_ns=timestamp,
                                     publication_monotonic_ns=published,age_ms=age,regions=len(boxes)))
            stream.flush()
            summaries.append(dict(repeat=repeat,age_ms=distribution(ages),region_count=distribution(counts)))
    connection.close()
    write_json(output/'summary.json',dict(repeats=summaries,
        scope='GStreamer source-buffer PTS mapped to host monotonic clock, through host JPEG encode, timestamped loopback transport, latest-frame selection, decode, CUDA OCR, annotation, JPEG encode and shared-buffer publication.',
        excludes='Physical exposure timestamp is not independently calibrated; HTTP transfer and browser rendering excluded. Translation disabled. This instrumented sender differs from the production unframed JPEG sender.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['sender','consumer'])
    parser.add_argument('--port',type=int,default=8555)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seconds',type=float,default=20)
    args=parser.parse_args()
    if args.mode=='sender': sender(args.port)
    else: consumer(args)
