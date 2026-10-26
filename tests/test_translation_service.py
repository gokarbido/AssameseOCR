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
            service.start()
            service.submit(['নমস্কাৰ'])
            self.assertEqual(service.snapshot(['নমস্কাৰ']), ([None], None))
            self.assertIsNone(service.thread)
        finally:
            service.stop()
        provider.assert_not_called()

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
