import numpy as np
import cv2 as cv
import os
from paddleocr import PaddleOCR


class TextDetection:
    def __init__(self, det_model_dir: str):
        self.ocr = PaddleOCR(text_detection_model_dir=det_model_dir)

    def detect_text_coordinates(self, image_path: str) -> np.ndarray:
        """
        Detects the bounding box coordinates of text in an image.

        Args:
            image_path (str): The file path to the input image for text detection.

        Returns:
            np.ndarray: A 4D array representing the bounding boxes of detected text with 
            dimensions (batch_size, num_bounding_boxes, num_points_per_box, 2). 
            Each bounding box consists of 4 points in the order: [lt, rt, rb, lb].
        """
        bounding_boxes = self.ocr.ocr(image_path, rec=False, cls=False, det=True)
        if isinstance(bounding_boxes, list):
            bounding_boxes = np.array(bounding_boxes)
        return bounding_boxes
    
    @staticmethod
    def sort_bounding_boxes(batch_bounding_boxes):
        """Sorts bounding boxes primarily by y-coordinate and then by x-coordinate.

        Args:
            batch_bounding_boxes (list): Bounding boxes coordinates of text.

        Returns:
            list: Sorted list of bounding boxes coordinates.
        """
        sorted_sorted_bounding_boxes = []
        for bounding_boxes in batch_bounding_boxes:
            sorted_bounding_boxes = sorted(bounding_boxes, key=lambda k: (k[0][1], k[0][0]))
            sorted_sorted_bounding_boxes.append(sorted_bounding_boxes)
        return sorted_sorted_bounding_boxes

    def draw_bounding_box_and_save(self, image, bounding_boxes, output_image_path: str):
        """Draws bounding boxes around text and saves the image.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.
            bounding_boxes (list): Bounding boxes coordinates.
            output_image_path (str): Path to save the output image.
        """
        bounding_boxes_batch = np.array(bounding_boxes, dtype=np.float32)
        for bounding_boxes in bounding_boxes_batch:
            for index, box in enumerate(bounding_boxes):
                x, y, w, h = cv.boundingRect(box)
                rectangular_box = cv.minAreaRect(box)
                coordinates = np.int32(cv.boxPoints(rectangular_box))
                cv.drawContours(image, [coordinates], -1, (0, 0, 255), 3, cv.LINE_AA)
                cv.putText(image, str(index + 1), (x + w // 4, y + h // 4),
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2, cv.LINE_AA)
            
            cv.imwrite(output_image_path, image)

# Example usage:
if __name__ == "__main__":
    det_model_dir = '../model/det_model/ch_PP-OCRv3_det_infer/'
    image_path = '../japanese-handwriting-images/image-01.jpeg'
    output_image_path = 'detect_text_example_output.jpg'

    # Initialize detector and detect text coordinates
    detector = TextDetection(det_model_dir)
    bounding_boxes = detector.detect_text_coordinates(image_path)
    sorted_boxes = TextDetection.sort_bounding_boxes(bounding_boxes)
    
    # Read the input image, draw bounding boxes and save the result
    image = cv.imread(image_path)
    detector.draw_bounding_box_and_save(image.copy(), sorted_boxes, output_image_path)
