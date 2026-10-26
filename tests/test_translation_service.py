"""Offline worker behavior with injected providers; no model or network calls."""
import threading
import unittest
from unittest.mock import Mock, patch

from translation_service import TranslationService


class TranslationServiceTests(unittest.TestCase):
    def wait_for(self, service, predicate):
        with service.condition:
            self.assertTrue(service.condition.wait_for(predicate, timeout=2),
                            'Translation worker did not publish the expected state')

    def test_cache_matches_source_and_suppresses_duplicate_requests(self):
        provider = Mock(side_effect=lambda text: {'অসমীয়া': 'Assamese', 'নমস্কাৰ': 'Hello'}[text])
        service = TranslationService(enabled=True, provider=provider, interval=0)
        try:
            service.start()
            service.submit([' অসমীয়া ', 'নমস্কাৰ', 'অসমীয়া', ''])
            self.wait_for(service, lambda: len(service.cache) == 2)
            service.submit(['অসমীয়া', 'নমস্কাৰ', 'অসমীয়া'])
            translations, error = service.snapshot(['নমস্কাৰ', 'নতুন', ' অসমীয়া '])
            self.assertEqual(translations, ['Hello', None, 'Assamese'])
            self.assertIsNone(error)
        finally:
            service.stop()
        self.assertEqual(provider.call_count, 2)

    def test_in_flight_result_never_attaches_to_new_source(self):
        old_started, finish_old = threading.Event(), threading.Event()
        new_started, finish_new = threading.Event(), threading.Event()

        def provider(text):
            if text == 'আগৰ':
                old_started.set()
                if not finish_old.wait(timeout=2):
                    raise RuntimeError('Test did not release the old request')
                return 'Previous'
            new_started.set()
            if not finish_new.wait(timeout=2):
                raise RuntimeError('Test did not release the new request')
            return 'Current'

        service = TranslationService(enabled=True, provider=provider, interval=0)
        try:
            service.start()
            service.submit(['আগৰ'])
            self.assertTrue(old_started.wait(timeout=2))
            service.submit(['এতিয়া'])
            finish_old.set()
            self.assertTrue(new_started.wait(timeout=2))
            self.assertEqual(service.snapshot(['এতিয়া'])[0], [None])
            self.assertEqual(service.snapshot(['আগৰ'])[0], ['Previous'])
            finish_new.set()
            self.wait_for(service, lambda: 'এতিয়া' in service.cache)
            self.assertEqual(service.snapshot(['এতিয়া'])[0], ['Current'])
        finally:
            finish_old.set()
            finish_new.set()
            service.stop()

    def test_failure_is_visible_then_retried_after_backoff(self):
        provider = Mock(side_effect=[RuntimeError('Local translation model is missing'), 'Hello'])
        service = TranslationService(enabled=True, provider=provider, interval=0, retry_delay=30)
        with patch('translation_service.time.monotonic', return_value=100.0) as clock:
            try:
                with self.assertLogs(level='WARNING'):
                    service.start()
                    service.submit(['নমস্কাৰ'])
                    self.wait_for(service, lambda: service.error is not None)
                    translations, error = service.snapshot(['নমস্কাৰ'])
                    self.assertEqual(translations, [None])
                    self.assertEqual(error, 'Local translation model is missing')
                    # Repeated frames cannot bypass the retry deadline.
                    service.submit(['নমস্কাৰ'])
                    service.submit(['নমস্কাৰ'])
                    self.assertEqual(provider.call_count, 1)
                    clock.return_value = 131.0
                    with service.condition:
                        service.condition.notify_all()
                    self.wait_for(service, lambda: 'নমস্কাৰ' in service.cache)
                    self.assertEqual(service.snapshot(['নমস্কাৰ']), (['Hello'], None))
            finally:
                service.stop()
        self.assertEqual(provider.call_count, 2)

    def test_disabled_translation_never_starts_provider(self):
        provider = Mock(return_value='Hello')
        service = TranslationService(enabled=False, provider=provider)
        try:
            with self.assertLogs(level='INFO') as logs:
                service.start()
                service.submit(['নমস্কাৰ'])
                self.assertEqual(service.snapshot(['নমস্কাৰ']), ([None], None))
                self.assertIsNone(service.thread)
            self.assertIn('Offline translation disabled.', logs.output[0])
        finally:
            service.stop()
        provider.assert_not_called()

    def test_success_logs_source_output_and_processing_time(self):
        service = TranslationService(enabled=True, provider=lambda text: 'Hello', interval=0)
        with patch('translation_service.time.perf_counter', side_effect=[10.0, 10.125]):
            with self.assertLogs(level='INFO') as logs:
                try:
                    service.start()
                    service.submit(['নমস্কাৰ'])
                    self.wait_for(service, lambda: 'নমস্কাৰ' in service.cache)
                finally:
                    service.stop()
        output = '\n'.join(logs.output)
        self.assertIn('Offline translation enabled.', output)
        self.assertIn('waiting for recognized text', output)
        self.assertIn("input='নমস্কাৰ'; output='Hello'; processing_ms=125.0", output)

    @patch('translation_service.OfflineTranslator')
    def test_model_loading_time_is_separate_from_translation_time(self, constructor):
        constructor.return_value.translate.return_value = 'Hello'
        service = TranslationService(enabled=True, interval=0)
        with patch('translation_service.time.perf_counter', side_effect=[10.0, 10.2, 10.3, 10.35]):
            with self.assertLogs(level='INFO') as logs:
                try:
                    service.start()
                    service.submit(['নমস্কাৰ'])
                    self.wait_for(service, lambda: 'নমস্কাৰ' in service.cache)
                finally:
                    service.stop()
        constructor.assert_called_once_with()
        output = '\n'.join(logs.output)
        self.assertIn('Offline translation model loading.', output)
        self.assertIn('model ready; load_ms=200.0', output)
        self.assertIn('processing_ms=50.0', output)

    def test_cache_logs_only_when_the_source_set_changes(self):
        provider = Mock(side_effect=lambda text: {'অসমীয়া': 'Assamese', 'নমস্কাৰ': 'Hello'}[text])
        service = TranslationService(enabled=True, provider=provider, interval=0)
        with self.assertLogs(level='INFO') as logs:
            try:
                service.start()
                service.submit(['অসমীয়া', 'নমস্কাৰ'])
                self.wait_for(service, lambda: len(service.cache) == 2)
                for _ in range(20):
                    service.submit(['নমস্কাৰ', 'অসমীয়া', 'নমস্কাৰ'])
                self.assertFalse(any('cache hit' in message for message in logs.output))
                service.submit([])
                service.submit(['অসমীয়া', 'নমস্কাৰ'])
                for _ in range(20):
                    service.submit(['অসমীয়া', 'নমস্কাৰ'])
            finally:
                service.stop()
        self.assertEqual(sum('cache hit' in message for message in logs.output), 2)
        self.assertEqual(provider.call_count, 2)

    def test_failure_logs_elapsed_time_and_retry_delay(self):
        provider = Mock(side_effect=RuntimeError('Model inference failed'))
        service = TranslationService(enabled=True, provider=provider, retry_delay=30)
        with patch('translation_service.time.perf_counter', side_effect=[10.0, 10.05]):
            with self.assertLogs(level='WARNING') as logs:
                try:
                    service.start()
                    service.submit(['নমস্কাৰ'])
                    self.wait_for(service, lambda: service.error is not None)
                finally:
                    service.stop()
        output = '\n'.join(logs.output)
        self.assertIn("input='নমস্কাৰ'; phase=translation; elapsed_ms=50.0", output)
        self.assertIn('retry_in_s=30.0; error=Model inference failed', output)

    def test_failed_region_does_not_block_other_regions(self):
        def provider(text):
            if text == 'bad':
                raise RuntimeError('Region is too long')
            return 'Hello'

        service = TranslationService(enabled=True, provider=provider, interval=0, retry_delay=30)
        try:
            with self.assertLogs(level='WARNING'):
                service.start()
                service.submit(['bad', 'good'])
                self.wait_for(service, lambda: 'good' in service.cache)
                self.assertEqual(service.snapshot(['bad', 'good']), ([None, 'Hello'], 'Region is too long'))
                self.assertEqual(service.snapshot(['good']), (['Hello'], None))
        finally:
            service.stop()

    def test_invalid_provider_result_is_reported_without_caching(self):
        service = TranslationService(enabled=True, provider=lambda text: ' ', retry_delay=30)
        try:
            with self.assertLogs(level='WARNING'):
                service.start()
                service.submit(['নমস্কাৰ'])
                self.wait_for(service, lambda: service.error is not None)
                self.assertEqual(service.snapshot(['নমস্কাৰ'])[0], [None])
                self.assertIn('empty result', service.error)
        finally:
            service.stop()


if __name__ == '__main__':
    unittest.main()
