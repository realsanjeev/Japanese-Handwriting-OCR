from pydantic import BaseModel
from typing import List, Tuple, Any
import numpy as np

class BoundingBox(BaseModel):
    """
    Represents a detected bounding box.
    Points are ordered: [lt, rt, rb, lb].
    """
    points: List[Tuple[float, float]]
    
    @property
    def check_valid(self) -> bool:
        return len(self.points) == 4

class TextSegment(BaseModel):
    """
    Represents a segment of text in the image.
    """
    id: int
    bounding_box: BoundingBox
    text: str = ""
    # We don't store the image crop in the model to avoid serialization issues,
    # but the pipeline will handle it.

class OCRResult(BaseModel):
    """
    Final result of the OCR process.
    """
    segments: List[TextSegment]
    output_image_path: str = ""
    output_json_path: str = ""
    
    def to_dict(self):
        # Helper for legacy JSON format compatibility if needed
        return {str(seg.id): seg.text for seg in self.segments}
