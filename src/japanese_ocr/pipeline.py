import cv2 as cv
import os
import logging
from typing import Optional
from .detection import TextDetector
from .recognition import TextRecognizer
from .processing import crop_box
from .domain import OCRResult, TextSegment
from .config import settings
from .utils import save_to_json, draw_boxes

logger = logging.getLogger(__name__)

class OCRPipeline:
    def __init__(self):
        logger.info("Initializing OCR Pipeline...")
        self.detector = TextDetector()
        self.recognizer = TextRecognizer()
        
    def process_image(self, image_path: str) -> Optional[OCRResult]:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
            
        logger.info(f"Processing image: {image_path}")
        image = cv.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
            
        # 1. Detection
        logger.info("Running detection...")
        boxes = self.detector.detect(image)
        logger.info(f"Detected {len(boxes)} text boxes.")
        
        segments = []
        
        # 2. Recognition Loop
        for i, box in enumerate(boxes):
            # Crop
            cropped = crop_box(image, box.points, offset=0)
            
            # PaddleOCR handles its own pre-processing (resize, norm), so we pass raw crop.
            text = self.recognizer.predict(cropped)
            
            segment = TextSegment(
                id=i + 1,
                bounding_box=box,
                text=text
            )
            segments.append(segment)
            
        # 3. Compile Result
        result = OCRResult(segments=segments)
        
        # 4. Save Outputs
        os.makedirs(settings.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # JSON
        json_path = os.path.join(settings.output_dir, f"{base_name}.json")
        save_to_json(result.to_dict(), json_path)
        
        # Annotated Image
        annotated_img = draw_boxes(image, segments)
        img_path = os.path.join(settings.output_dir, f"{base_name}.jpg")
        cv.imwrite(img_path, annotated_img)
        
        result.output_image_path = img_path
        result.output_json_path = json_path
        logger.info(f"Finished processing. Saved to {settings.output_dir}")
        
        return result
