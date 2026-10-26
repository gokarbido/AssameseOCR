"""Cached offline Assamese-to-English translation, separate from the camera."""
from collections import OrderedDict
import logging
import threading
import time

from offline_translation import OfflineTranslator


class TranslationService:
    def __init__(self, enabled=False, provider=None, interval=0.5, retry_delay=30):
        self.enabled = enabled
        self.provider = provider
        self.model = None
        self.interval, self.retry_delay = interval, retry_delay
        self.condition = threading.Condition()
        self.cache = OrderedDict()
        self.failures = OrderedDict()
        self.desired = ()
        self.error = None
        self.stopped = False
        self.thread = None

    def start(self):
        if self.enabled:
            logging.info('Offline translation enabled.')
            logging.info('Offline translation waiting for recognized text.')
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        else:
            logging.info('Offline translation disabled.')

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=1)

    def submit(self, texts):
        if self.enabled:
            with self.condition:
                desired = tuple(dict.fromkeys(text.strip() for text in texts if text.strip()))
                changed = frozenset(desired) != frozenset(self.desired)
                cached = [(text, self.cache[text]) for text in desired if text in self.cache] if changed else []
                self.desired = desired
                self.condition.notify_all()
            for source, english in cached:
                logging.info('Offline translation cache hit: input=%r; output=%r', source, english)
            if changed and not desired:
                logging.info('Offline translation waiting for recognized text.')

    def snapshot(self, texts):
        with self.condition:
            translated = [self.cache.get(text.strip()) if self.enabled else None for text in texts]
            error = next((self.failures[text.strip()][1] for text, value in zip(texts, translated)
                          if self.enabled and value is None and text.strip() in self.failures), None)
            return translated, error

    def _run(self):
        next_attempt = 0
        while True:
            with self.condition:
                while not self.stopped:
                    now = time.monotonic()
                    pending = [text for text in self.desired if text not in self.cache]
                    text = next((text for text in pending if self.failures.get(text, (0,))[0] <= now), None)
                    delay = next_attempt - now
                    if text is not None and delay <= 0:
                        break
                    retry = min((self.failures.get(text, (0,))[0] for text in pending), default=0) - now
                    self.condition.wait(timeout=max(delay, retry, 0.01) if pending else None)
                if self.stopped:
                    return
            started = time.perf_counter()
            phase = 'translation'
            try:
                if self.provider is None and self.model is None:
                    phase = 'model loading'
                    logging.info('Offline translation model loading.')
                    self.model = OfflineTranslator()
                    logging.info('Offline translation model ready; load_ms=%.1f',
                                 (time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    phase = 'translation'
                translated = (self.provider or self.model.translate)(text)
                if not isinstance(translated, str) or not translated.strip():
                    raise RuntimeError('Translation service returned an empty result.')
                logging.info('Offline translation: input=%r; output=%r; processing_ms=%.1f',
                             text, translated, (time.perf_counter() - started) * 1000)
                with self.condition:
                    self.cache[text] = translated
                    self.cache.move_to_end(text)
                    if len(self.cache) > 256:
                        self.cache.popitem(last=False)
                    self.failures.pop(text, None)
                    self.error = None
                    next_attempt = time.monotonic() + self.interval
                    self.condition.notify_all()
            except Exception as exc:
                logging.warning('Offline translation failed: input=%r; phase=%s; elapsed_ms=%.1f; '
                                'retry_in_s=%.1f; error=%s',
                                text, phase, (time.perf_counter() - started) * 1000,
                                self.retry_delay, exc)
                with self.condition:
                    self.error = str(exc)
                    self.failures[text] = (time.monotonic() + self.retry_delay, self.error)
                    self.failures.move_to_end(text)
                    if len(self.failures) > 256:
                        self.failures.popitem(last=False)
                    next_attempt = time.monotonic() + self.interval
                    self.condition.notify_all()
