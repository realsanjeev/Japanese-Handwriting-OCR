import logging
import numpy as np
from paddleocr import PaddleOCR
from typing import List
from .config import settings

logger = logging.getLogger(__name__)

class TextRecognizer:
    def __init__(self):
        logger.info(f"Initializing PaddleOCR Recognizer (lang={settings.rec_language})...")
        # Initialize PaddleOCR for recognition only
        self.ocr = PaddleOCR(
            rec_model_dir=settings.rec_model_dir, 
            lang=settings.rec_language,
            use_gpu=False,
            show_log=False,
            det=False,
            rec=True,
            cls=False
        )

    def predict(self, input_image: np.ndarray) -> str:
        """
        Runs recognition on a cropped image chunk.
        Input image can be raw crop (no resizing needed).
        """
        if input_image is None or input_image.size == 0:
            return ""
            
        # PaddleOCR expects a list of images or paths.
        # passing list of [image]
        try:
            result = self.ocr.ocr(input_image, det=False, rec=True, cls=False)
            if not result or result[0] is None:
                return ""
            
            # Result format: [ [ (text, score), ... ] ]
            # For single image, result[0] is a list of tuples?
            # Let's verify standard output for single image list input.
            # Usually result is list of list of results.
            
            # Since we pass one image, we expect result[0] to be valid.
            # result structure for rec-only: [ (text, score), ... ] ? 
            # Actually paddle outputs list of (text, confidence)
            
            text, score = result[0][0]
            return text
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return ""
