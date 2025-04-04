import json
from typing import Any
import numpy as np
import cv2 as cv

def save_to_json(data: Any, save_path: str) -> None:
    """
    Save data to a JSON file.
    """
    with open(save_path, 'wb') as fp:
        fp.write(json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8"))

def draw_boxes(image: np.ndarray, segments: list) -> np.ndarray:
    """
    Draw bounding boxes and IDs on the image.
    """
    annotated_image = image.copy()
    for segment in segments:
        bbox = segment.bounding_box
        points = np.array(bbox.points, dtype=np.int32)
        
        # Draw contour
        cv.drawContours(annotated_image, [points], -1, (0, 0, 255), 3, cv.LINE_AA)
        
        # Calculate center for text
        x, y, w, h = cv.boundingRect(points)
        cv.putText(annotated_image, str(segment.id), (x + w // 4, y + h // 4),
                cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2, cv.LINE_AA)
                
    return annotated_image
