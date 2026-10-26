"""Geometric reading order for horizontal Assamese text regions."""
import numpy as np


def sort_detections(boxes, scores):
    """Group by line, then order lines top-down and regions left-to-right."""
    if len(boxes) < 2:
        return boxes, scores
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    centers = (boxes[:, 1] + boxes[:, 3]) / 2
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1)
    rows = []
    for index in sorted(range(len(boxes)), key=lambda i: (centers[i], *boxes[i], i)):
        matches = []
        for row_index, row in enumerate(rows):
            center, height = float(np.median(centers[row])), float(np.median(heights[row]))
            overlap = min(boxes[index, 3], center + height / 2) - max(boxes[index, 1], center - height / 2)
            distance = abs(centers[index] - center)
            # Both conditions prevent a tall box from joining separate lines.
            if overlap >= 0.5 * min(heights[index], height) and distance <= 0.5 * min(heights[index], height):
                matches.append((distance, center, row_index))
        if matches:
            rows[min(matches)[2]].append(index)
        else:
            rows.append([index])
    rows.sort(key=lambda row: (float(np.median(centers[row])), min(boxes[i, 0] for i in row)))
    order = [i for row in rows for i in sorted(row, key=lambda i: (*boxes[i], i))]
    return boxes[order], scores[order]
