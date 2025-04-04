import numpy as np
from paddleocr import PaddleOCR
from typing import List
from .domain import BoundingBox
from .config import settings

class TextDetector:
    def __init__(self):
        # Initialize PaddleOCR with settings
        # We pass use_gpu=False as per original script default, but could be configurable
        self.ocr = PaddleOCR(
            text_detection_model_dir=settings.det_model_dir,
            det_db_thresh=settings.det_db_thresh,
            det_db_box_thresh=settings.det_db_box_thresh,
            use_gpu=False,
            show_log=False
        )

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detects text bounding boxes in the image.
        """
        # PaddleOCR expects path or array. Passing array avoids disk I/O.
        # rec=False, cls=False to only do detection
        result = self.ocr.ocr(image, rec=False, cls=False, det=True)
        
        boxes = []
        if not result or result[0] is None:
            return boxes
            
        # Result structure is roughly: [ [ [ [x,y], ... ], ... ] ]
        # Flatten the batch dimension if present
        raw_boxes = result[0]
        
        for box_points in raw_boxes:
            # box_points is a list of 4 [x, y] lists
            # Convert to list of tuples for our domain model
            points_tuple = [tuple(pt) for pt in box_points]
            boxes.append(BoundingBox(points=points_tuple))
            
        return self._sort_boxes(boxes)

    @staticmethod
    def _sort_boxes(boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Sorts boxes primarily by Y (top to bottom), then X (left to right).
        """
        # Sort key: top-left Y, then top-left X
        # points[0] is typically top-left
        return sorted(boxes, key=lambda b: (b.points[0][1], b.points[0][0]))
