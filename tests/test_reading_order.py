"""Reading order must survive detection confidence order and vertical jitter."""
import unittest
from unittest.mock import Mock

import numpy as np

from app import LiveTextDetectionApp
from reading_order import sort_detections


class ReadingOrderTests(unittest.TestCase):
    def test_rows_then_columns_keep_scores_attached(self):
        boxes = np.array([[100, 60, 130, 80], [100, 10, 130, 30],
                          [0, 64, 30, 84], [0, 14, 30, 34]])
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        ordered, confidence = sort_detections(boxes, scores)
        np.testing.assert_array_equal(ordered, boxes[[3, 1, 2, 0]])
        np.testing.assert_array_equal(confidence, scores[[3, 1, 2, 0]])

    def test_tall_region_cannot_bridge_two_rows(self):
        boxes = np.array([[0, 40, 30, 60], [100, 0, 130, 80], [50, 10, 80, 30]])
        ordered, _ = sort_detections(boxes, np.arange(3))
        self.assertEqual(ordered.tolist().index(boxes[2].tolist()), 0)
        self.assertGreater(ordered.tolist().index(boxes[0].tolist()), 0)

    def test_mixed_heights_share_the_same_line(self):
        boxes = np.array([[70, 0, 90, 40], [10, 10, 30, 30]])
        ordered, _ = sort_detections(boxes, np.arange(2))
        np.testing.assert_array_equal(ordered, boxes[[1, 0]])

    def test_empty_single_and_duplicate_boxes_are_preserved(self):
        for boxes in (np.array([]), np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4], [1, 2, 3, 4]])):
            with self.subTest(count=len(boxes)):
                ordered, scores = sort_detections(boxes, np.arange(len(boxes)))
                np.testing.assert_array_equal(ordered, boxes)
                np.testing.assert_array_equal(scores, np.arange(len(boxes)))

    def test_ocr_results_follow_sorted_regions(self):
        app = LiveTextDetectionApp.__new__(LiveTextDetectionApp)
        app.frame_counter, app.log_interval = 0, 30
        app.detector, app.recognizer = Mock(), Mock()
        # Recognition uses a pixel marker in each region to reveal any mismatch.
        boxes = np.array([[30, 2, 40, 12], [2, 3, 12, 13], [1, 24, 11, 34]])
        frame = np.zeros((40, 50, 3), dtype=np.uint8)
        for value, (x1, y1, x2, y2) in enumerate(boxes, 1):
            frame[y1:y2, x1:x2] = value
        app.detector.detect_one.return_value = (boxes, np.array([0.3, 0.8, 0.6]))
        app.recognizer.recognize.side_effect = lambda image: str(image[0, 0, 0])
        ordered, scores, texts = app.process_frame(frame)
        np.testing.assert_array_equal(ordered, boxes[[1, 0, 2]])
        np.testing.assert_array_equal(scores, [0.8, 0.3, 0.6])
        self.assertEqual(texts, ['2', '1', '3'])


if __name__ == '__main__':
    unittest.main()
