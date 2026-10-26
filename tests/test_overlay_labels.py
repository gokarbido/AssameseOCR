"""Large labels stay bounded and English retains a scalable font."""
import unittest
from unittest.mock import patch

import numpy as np
import cv2
from PIL import Image, ImageDraw

from app import LiveTextDetectionApp
from overlay_labels import font_size, label_layout, place_label


class OverlayTests(unittest.TestCase):
    def test_larger_fonts_scale_with_video_resolution(self):
        self.assertEqual(font_size(1080), 48)
        self.assertGreater(font_size(360), 24)

    def test_long_label_wraps_and_stays_inside_width(self):
        draw = ImageDraw.Draw(Image.new('RGB', (320, 240)))
        layout = label_layout(draw, 'Large English translation ' * 20, (240, 320, 3), 'en')
        self.assertLessEqual(layout['width'], 320)
        self.assertLessEqual(len(layout['text'].splitlines()), 2)
        self.assertIn('…', layout['text'])
        self.assertGreaterEqual(getattr(layout['font'], 'size', 0), 28)

    def test_labels_near_edges_do_not_change_the_bounding_box(self):
        app = LiveTextDetectionApp.__new__(LiveTextDetectionApp)
        app.font_path = None
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        boxes = np.array([[400, 0, 639, 40]])
        with patch('app.cv2.rectangle', wraps=cv2.rectangle) as rectangle:
            rendered = app.draw_detections(frame, boxes, np.array([0.9]), ['Source'], ['English'])
        rectangle.assert_called_once()
        self.assertEqual(rectangle.call_args.args[1:3], ((400, 0), (639, 40)))
        self.assertEqual(rectangle.call_args.args[4], 2)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertTrue(np.any(rendered))

    def test_neighboring_label_blocks_do_not_overlap_when_space_is_available(self):
        occupied = []
        boxes = [(1400, 0, 1919, 65), (1100, 240, 1850, 330)]
        first = place_label(boxes[0], (1080, 1920, 3), 400, 120, occupied, boxes)
        second = place_label(boxes[1], (1080, 1920, 3), 600, 140, occupied, boxes)
        self.assertEqual(first, (1400, 73))
        self.assertGreaterEqual(second[1], 330)
        self.assertEqual(len(occupied), 2)
        self.assertTrue(occupied[0][3] <= occupied[1][1] or occupied[0][2] <= occupied[1][0]
                        or occupied[1][3] <= occupied[0][1] or occupied[1][2] <= occupied[0][0])

    def test_oversized_or_crowded_blocks_are_not_drawn_over_other_labels(self):
        occupied = []
        self.assertIsNone(place_label((0, 0, 50, 50), (100, 100, 3), 80, 120, occupied, []))
        self.assertEqual(occupied, [])
        self.assertIsNotNone(place_label((10, 10, 20, 20), (100, 100, 3), 80, 80, occupied, []))
        self.assertIsNone(place_label((10, 10, 20, 20), (100, 100, 3), 80, 80, occupied, []))
        self.assertEqual(len(occupied), 1)


if __name__ == '__main__':
    unittest.main()
