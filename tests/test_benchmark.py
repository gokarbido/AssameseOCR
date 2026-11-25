"""Check evidence integrity, image replay and metric definitions without a GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from benchmark import (box_iou, distribution, edit_distance, evaluate_reviewed,
                       load_manifest, match_boxes, normalize, run_frame)

spec = importlib.util.spec_from_file_location('record_jetson_experiment', Path(__file__).resolve().parents[1] / 'scripts/record_jetson_experiment.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class MetricTests(unittest.TestCase):
    def test_unicode_normalization_and_edit_counts(self):
        self.assertEqual(normalize('  অসমীয়া\n পাঠ  '), 'অসমীয়া পাঠ')
        self.assertEqual(edit_distance('abc', 'adc'), 1)
        self.assertEqual(edit_distance('', 'abc'), 3)
        self.assertEqual(edit_distance('abc', ''), 3)

    def test_matching_is_one_to_one_and_thresholded(self):
        gt = [[0, 0, 20, 20]]
        predicted = [[0, 0, 20, 20], [1, 1, 19, 19], [50, 50, 60, 60]]
        self.assertEqual(match_boxes(gt, predicted), [(0, 0)])
        self.assertEqual(box_iou(gt[0], predicted[2]), 0)
        self.assertEqual(match_boxes([], predicted), [])

    def test_empty_distributions_stay_unknown(self):
        self.assertIsNone(distribution([]))
        self.assertEqual(distribution([1, 2, 3])['p50'], 2)

    def test_unreviewed_prompted_text_is_never_scored(self):
        pipeline = Mock()
        result = evaluate_reviewed(pipeline, [dict(id='a', requested_text=['অসমীয়া'], annotation_status='unreviewed')], {}, 0)
        self.assertEqual(result['status'], 'not_scored')
        self.assertNotIn('crop_cer_percent', result)
        pipeline.recognizer.recognize.assert_not_called()

    def test_real_tegrastats_units_and_missing_sensors(self):
        raw = 'RAM 3320/7620MB CPU [1%@900] GR3D_FREQ 37%@[612] CPU@48.5C GPU@49C VDD_IN 6282mW/6028mW'
        self.assertEqual(host.parse_tegrastats(raw), dict(gpu_percent=37, system_ram_mb=3320, module_power_w=6.282, max_cpu_gpu_c=49))
        self.assertEqual(host.parse_tegrastats('RAM 1024/7620MB'), {'system_ram_mb': 1024})
        self.assertEqual(host.parse_tegrastats('failed to read sensor'), {})


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        cv2.imwrite(str(self.root / 'image.png'), np.full((100, 200, 3), 120, dtype=np.uint8))
        self.sample = dict(id='test', image='image.png', annotation_status='unreviewed')

    def manifest(self, samples):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(dict(samples=samples)))
        return path

    def pipeline(self):
        pipeline = Mock()
        pipeline.detector.detect_one.return_value = (np.array([[2, 2, 20, 20], [40, 2, 60, 20]]), np.array([0.9, 0.8]))
        pipeline.recognizer.recognize.side_effect = ['অসমীয়া', RuntimeError('failed crop')]
        pipeline.draw_detections.side_effect = lambda frame, *args: frame
        return pipeline

    def test_manifest_blocks_paths_duplicates_and_fake_review(self):
        for samples in ([dict(self.sample, image='../outside.png')], [self.sample, self.sample],
                        [dict(self.sample, id='../outside')],
                        [dict(self.sample, annotation_status='reviewed')]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                load_manifest(self.manifest(samples))

    def test_replay_preserves_failed_crops_and_records_actual_dimensions(self):
        _, samples = load_manifest(self.manifest([self.sample]))
        timing, prediction, encoded = run_frame(self.pipeline(), samples[0], 100)
        self.assertEqual(prediction['source_width'], 200)
        self.assertEqual(prediction['input_width'], 100)
        self.assertEqual(prediction['texts'], ['অসমীয়া', ''])
        self.assertEqual(prediction['recognition_errors'], ['failed crop'])
        self.assertEqual(len(prediction['boxes']), len(prediction['texts']))
        self.assertTrue(encoded.tobytes().startswith(b'\xff\xd8'))
        self.assertGreaterEqual(timing['processing_ms'], timing['detect_ms'])

    def test_reviewed_metrics_account_for_false_positives_and_missed_regions(self):
        sample = dict(self.sample, annotation_status='reviewed', reviewer='Independent reviewer',
                      regions=[dict(box=[0, 0, 40, 40], text='অসমীয়া'), dict(box=[100, 0, 140, 40], text='পাঠ')])
        _, samples = load_manifest(self.manifest([sample]))
        pipeline = Mock()
        pipeline.recognizer.recognize.side_effect = ['অসমীয়া', 'পাঠ']
        predictions = {'test': dict(boxes=[[0, 0, 40, 40], [50, 50, 70, 70]], texts=['অসমীয়া', 'ভুল'])}
        result = evaluate_reviewed(pipeline, samples, predictions, 0)
        self.assertEqual(result['detection']['precision'], 0.5)
        self.assertEqual(result['detection']['recall'], 0.5)
        self.assertEqual(result['spotting']['hmean'], 0.5)
        self.assertEqual(result['crop_cer_percent'], 0)
        self.assertEqual(result['crop_exact_region_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
