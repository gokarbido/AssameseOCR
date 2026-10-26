import cv2
import numpy as np
import onnxruntime as ort
from typing import Sequence, Union, Tuple, List

# Define custom types for clarity
SizeType = Union[int, Tuple[int, int]] # Example: 640 or (640, 640)
BoxesType = np.ndarray # Array of bounding boxes (x1, y1, x2, y2 format after post-processing)
ScoresType = np.ndarray # Array of confidence scores

# Define custom exceptions if they are not already defined in your project
class InvalidModelParamException(Exception):
    pass

class InvalidInputException(Exception):
    pass

# --- Helper functions (can be static methods in class or outside) ---

def _letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114), auto: bool = True, scaleFill: bool = False, scaleup: bool = True):
    """
    Resizes and pads an image to a new shape, maintaining aspect ratio (letterbox).
    Returns padded image, ratio, and padding amounts (left_pad, top_pad).
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape) # (target_height, target_width)

    # Scale ratio (new / current)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # (width, height) after scaling
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding to target shape

    if auto:
        pass # No explicit stride adjustment for simplicity and exact new_shape match
    elif scaleFill:  # stretch (no padding)
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize if dimensions changed
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, r, (left, top) # return padded image, ratio, and actual (left_padding, top_padding)

# --- PlateDetector Class ---

class YoloObjectDetector:
    def __init__(
        self,
        size: SizeType = 640, # Can be 640 (for 640x640) or (W, H) tuple like (640, 480)
        batched: bool = True,
        path: str = None,
        providers: Sequence[str] = None,
        score_threshold: float = 0.25, # Adjusted for YOLOv8 common usage
        iou_threshold: float = 0.45,   # Adjusted for YOLOv8 common usage
        keep_top_k: int = 1024,        # Parameter from template, not directly used with cv2.dnn.NMSBoxes
        top_k: int = 256,              # Parameter from template, not directly used with cv2.dnn.NMSBoxes
        target_class_id: int = 0,      # New parameter to filter for a specific class ID
    ):
        """
        Initializes the PlateDetector.

        :param size: Target size for model input. Can be an int (for square e.g., 640 -> 640x640) or a tuple (width, height).
        :param batched: Whether the model supports batched input.
        :param path: Path to the ONNX model file (e.g., 'yolov11n.onnx').
        :param providers: ONNX Runtime execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        :param score_threshold: Confidence score threshold for detections.
        :param iou_threshold: IoU threshold for Non-Maximum Suppression.
        :param keep_top_k: Not directly used by `cv2.dnn.NMSBoxes` as it handles all detections above score_threshold.
        :param top_k: Not directly used by `cv2.dnn.NMSBoxes`.
        :param target_class_id: The class ID to filter for (e.g., 0 for 'license_plate'). Only BBoxes for this ID will be returned.
        """
        self._batched = batched
        
        # Determine model input shape based on 'size' parameter
        if isinstance(size, int):
            self._model_input_shape = (size, size) # (width, height)
        elif isinstance(size, tuple) and len(size) == 2:
            self._model_input_shape = size # (width, height)
        else:
            raise ValueError("Invalid 'size' parameter. Must be an int or a tuple (width, height).")

        self._session = self._create_session(path, providers)
        
        session_input = self._session.get_inputs()[0]
        self._input_name = session_input.name

        # Validate ONNX model's expected input shape
        # session_input.shape is typically (N, C, H, W)
        onnx_model_h = session_input.shape[2]
        onnx_model_w = session_input.shape[3]
        onnx_model_shape_wh = (onnx_model_w, onnx_model_h) # (W, H) from ONNX model

        # Compare with the detector's specified model_input_shape
        if onnx_model_shape_wh != self._model_input_shape:
            raise InvalidModelParamException(
                f'ONNX model\'s input shape is {onnx_model_shape_wh} (W, H), while detector '
                f'is configured for {self._model_input_shape} (W, H)! '
                'Change `size` param or use another model file.',
            )

        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._keep_top_k = keep_top_k # Retained for compatibility with template
        self._top_k = top_k           # Retained for compatibility with template
        self._target_class_id = target_class_id # Store the target class ID

        if not (0 <= self._score_threshold <= 1):
            raise InvalidModelParamException(
                f'Invalid score threshold: {self._score_threshold}, it should be in range [0..1]'
            )
        if not (0 <= self._iou_threshold <= 1):
            raise InvalidModelParamException(
                f'Invalid iou threshold: {self._iou_threshold}, it should be in range [0..1]'
            )

    @classmethod
    def _create_session(
        cls,
        path,
        providers,
    ):
        if path is None:
            raise InvalidModelParamException("Model path must be provided to load the ONNX model.")
        
        # Default providers if none are given
        if providers is None or len(providers) == 0:
            providers = ['CPUExecutionProvider'] # Fallback to CPU
        
        try:
            return ort.InferenceSession(path, providers=providers)
        except Exception as e:
            raise InvalidModelParamException(f"Failed to load ONNX model at '{path}': {e}")

    def detect_one(self, image: np.ndarray, fit_to_image: bool = False) -> Tuple[BoxesType, ScoresType]:
        """
        Detects plates in a single image.

        :param image: Input image as a NumPy array (HWC, BGR).
        :param fit_to_image: Whether to clip bounding boxes to image boundaries.
        :return: A tuple containing (boxes, scores) in (x1, y1, x2, y2) format.
        """
        if image is None:
            raise InvalidInputException('Image is None')
        
        # detect_batch returns list of (boxes, scores, class_ids) tuples
        result_batch = self.detect_batch(np.asarray([image]), fit_to_image)
        if result_batch:
            # Extract only boxes and scores from the first (and only) result
            boxes, scores, _ = result_batch[0]
            return boxes, scores
        return np.asarray([]), np.asarray([], dtype=np.float32) # Return empty if no detections

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        fit_to_image: bool = False,
    ) -> List[Tuple[BoxesType, ScoresType, List[int]]]:
        """
        Detects plates in a batch of images.

        :param images: A sequence of input images as NumPy arrays (HWC, BGR).
        :param fit_to_image: Whether to clip bounding boxes to image boundaries.
        :return: A list of tuples, each containing (boxes, scores, class_ids), where boxes are in (x1, y1, x2, y2) format.
        """
        if images is None or len(images) == 0:
            return []
        
        for i, image in enumerate(images):
            if image is None:
                raise InvalidInputException(f'Image with index {i} is None')
        
        if len(images) > 1 and not self._batched:
            raise InvalidInputException(
                f'Current detector is not batched, but images count is more than 1 (count is {len(images)}), '
                f'use `batched=True` param in detector\'s constructor to fix this'
            )
        
        # Preprocess images (applying letterbox, normalization, etc.)
        # _model_input_shape is (W, H), _letterbox expects (H, W) for new_shape.
        processed_images_tensor, original_dims_list, letterbox_ratios, letterbox_pads = self._preprocess_images(images)
        
        # Run inference
        outputs = self._session.run(None, {self._input_name: processed_images_tensor})
        
        results = []
        # YOLOv8-like models usually have a single output tensor (1, 84, 8400) or (N, 84, 8400) for batch
        raw_model_output = outputs[0]

        for i, original_image_dims in enumerate(original_dims_list):
            original_h, original_w = original_image_dims
            ratio = letterbox_ratios[i]
            pad_w, pad_h = letterbox_pads[i]

            # Extract predictions for the current image in the batch
            # If batched, raw_model_output[i] for each image. If not batched, it's just raw_model_output.
            current_image_predictions = raw_model_output[i:i+1] if self._batched else raw_model_output

            # Transpose and squeeze to get (num_boxes, features) -> (8400, 84)
            predictions_processed = np.squeeze(current_image_predictions).T

            boxes, scores, class_ids = self._postprocess_yolov8_output(
                predictions_processed,
                original_w, original_h,
                self._model_input_shape[0], self._model_input_shape[1], # model_input_w, model_input_h
                ratio, pad_w, pad_h,
                fit_to_image
            )
            results.append((boxes, scores, class_ids))
        
        return results

    def _preprocess_images(self, images: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, int]], List[float], List[Tuple[int, int]]]:
        processed_images_list = []
        original_dims_list = []
        ratios_list = []
        pads_list = []

        target_w, target_h = self._model_input_shape # (W, H)
        
        for image in images:
            original_h, original_w, _ = image.shape
            original_dims_list.append((original_h, original_w))

            # Apply letterbox padding. _letterbox expects (H, W) for new_shape.
            padded_image, ratio, (pad_w, pad_h) = _letterbox(image, new_shape=(target_h, target_w))
            ratios_list.append(ratio)
            pads_list.append((pad_w, pad_h))

            # Convert to RGB, normalize to [0, 1] then transpose to CHW
            rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
            normalized_image = rgb_image.astype(np.float32) / 255.0 # Normalization to [0,1]
            input_tensor = np.transpose(normalized_image, (2, 0, 1)) # HWC to CHW
            processed_images_list.append(input_tensor)

        # Stack into a batch (N, C, H, W)
        return np.stack(processed_images_list), original_dims_list, ratios_list, pads_list

    def _postprocess_yolov8_output(
        self,
        predictions: np.ndarray, # (num_boxes, 84) where 84 is (xywh + num_classes)
        original_w: int,
        original_h: int,
        model_input_w: int,
        model_input_h: int,
        ratio: float, # Scaling ratio from letterbox
        pad_w: int,   # Width padding from letterbox
        pad_h: int,   # Height padding from letterbox
        fit_to_image: bool = False,
    ) -> Tuple[BoxesType, ScoresType, List[int]]: # BoxesType here is (x1,y1,x2,y2)

        # Extract raw box coordinates and class scores
        boxes_raw = predictions[:, :4] # xywh (relative to model_input_w x model_input_h)
        class_scores_raw = predictions[:, 4:] # num_classes scores

        # Find the best class confidence and its ID for each prediction
        class_ids_all = np.argmax(class_scores_raw, axis=1)
        confidences_all = np.max(class_scores_raw, axis=1)

        # 1. Filter by score threshold AND target_class_id
        mask = (confidences_all > self._score_threshold) & (class_ids_all == self._target_class_id)
        
        boxes_filtered = boxes_raw[mask]
        confidences_filtered = confidences_all[mask]
        class_ids_filtered = class_ids_all[mask]

        if len(boxes_filtered) == 0:
            return np.asarray([]), np.asarray([], dtype=np.float32), []

        # 2. Convert xywh (center, width, height) to xyxy (top-left, bottom-right)
        # These are still relative to the padded model input size (e.g., 640x640)
        x_center, y_center, w, h = boxes_filtered[:, 0], boxes_filtered[:, 1], boxes_filtered[:, 2], boxes_filtered[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # 3. De-letterbox (remove padding and scale to original image dimensions)
        # Remove padding
        x1 -= pad_w
        y1 -= pad_h
        x2 -= pad_w
        y2 -= pad_h

        # Scale by the ratio to get back to original image dimensions
        x1 /= ratio
        y1 /= ratio
        x2 /= ratio
        y2 /= ratio

        # Combine into xyxy format for NMSBoxes
        final_boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) # This is now the final bbox format

        # 4. Apply Non-Maximum Suppression (NMS)
        # NMSBoxes expects boxes in [x1, y1, x2, y2] format
        indices = cv2.dnn.NMSBoxes(
            final_boxes_xyxy.tolist(), # Convert to list for NMSBoxes
            confidences_filtered.tolist(), # Convert to list
            self._score_threshold,
            self._iou_threshold
        )
        
        final_boxes_filtered_by_nms = []
        final_confidences = []
        final_class_ids = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes_filtered_by_nms.append(final_boxes_xyxy[i]) # Keep xyxy format
                final_confidences.append(confidences_filtered[i])
                final_class_ids.append(class_ids_filtered[i])

        final_boxes_filtered_by_nms = np.asarray(final_boxes_filtered_by_nms)
        final_confidences = np.asarray(final_confidences, dtype=np.float32)

        if len(final_boxes_filtered_by_nms) == 0:
            return np.asarray([]), np.asarray([], dtype=np.float32), []

        # 5. Apply fit_to_image (clipping directly on x1, y1, x2, y2)
        if fit_to_image:
            final_boxes_filtered_by_nms[:, 0] = np.clip(final_boxes_filtered_by_nms[:, 0], 0, original_w - 1)
            final_boxes_filtered_by_nms[:, 1] = np.clip(final_boxes_filtered_by_nms[:, 1], 0, original_h - 1)
            final_boxes_filtered_by_nms[:, 2] = np.clip(final_boxes_filtered_by_nms[:, 2], 0, original_w - 1)
            final_boxes_filtered_by_nms[:, 3] = np.clip(final_boxes_filtered_by_nms[:, 3], 0, original_h - 1)
            
            # Ensure x2 >= x1 and y2 >= y1 after clipping
            # This handles cases where clipping might make x2 < x1 or y2 < y1
            final_boxes_filtered_by_nms[:, 2] = np.maximum(final_boxes_filtered_by_nms[:, 2], final_boxes_filtered_by_nms[:, 0])
            final_boxes_filtered_by_nms[:, 3] = np.maximum(final_boxes_filtered_by_nms[:, 3], final_boxes_filtered_by_nms[:, 1])


        # Return in (x1, y1, x2, y2) format
        return final_boxes_filtered_by_nms.astype(np.int32), final_confidences, final_class_ids