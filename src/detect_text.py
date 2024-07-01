from paddleocr import PaddleOCR
import numpy as np
import cv2 as cv

class TextDetection:
    def __init__(self, det_model_dir):
        self.ocr = PaddleOCR(det_model_dir=det_model_dir)

    def detect_text_coordinates(self, image_path):
        """Detects bounding boxes coordinates of text.

        Args:
            image_path (str): Path to input image for text detection.

        Returns:
            list: List of bounding boxes coordinates, each box represented as a list of points.
        """
        bounding_boxes = self.ocr.ocr(image_path, rec=False, cls=False, det=True)
        return bounding_boxes
    
    @staticmethod
    def sort_bounding_boxes(bounding_boxes):
        """Sorts bounding boxes primarily by y-coordinate and then by x-coordinate.

        Args:
            bounding_boxes (list): Bounding boxes coordinates of text.

        Returns:
            list: Sorted list of bounding boxes coordinates.
        """
        sorted_bounding_boxes = sorted(bounding_boxes, key=lambda k: (k[0][1], k[0][0]))
        return sorted_bounding_boxes

    def draw_bounding_box_and_save(self, image, bounding_boxes, output_image_path):
        """Draws bounding boxes around text and saves the image.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.
            bounding_boxes (list): Bounding boxes coordinates.
            output_image_path (str): Path to save the output image.
        """
        formatted_bounding_boxes = [np.float32([[point] for point in box]) for box in bounding_boxes]

        for index, box in enumerate(formatted_bounding_boxes):
            x, y, w, h = cv.boundingRect(box)
            rectangular_box = cv.minAreaRect(box)
            coordinates = np.int0(cv.boxPoints(rectangular_box))
            cv.drawContours(image, [coordinates], 0, (0, 0, 255), 3, cv.LINE_AA)
            cv.putText(image, str(index + 1), (x + w // 4, y + h // 4),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2, cv.LINE_AA)
        
        cv.imwrite(output_image_path, image)

# Example usage:
if __name__ == "__main__":
    det_model_dir = '/home/sanjeev/Desktop/test/Japanese-Handwritten-OCR/model/det_model/ch_PP-OCRv3_det_infer/'
    image_path = '/home/sanjeev/Desktop/test/Japanese-Handwritten-OCR/demo.png'
    output_image_path = 'path_to_output_image.jpg'
    detector = TextDetection(det_model_dir)
    with open("XXXXXXXX.tst", "w") as fp:
        fp.write("This is the life")
    print("+"*32)
    bounding_boxes = detector.detect_text_coordinates(image_path)
    sorted_boxes = TextDetection.sort_bounding_boxes(bounding_boxes)
    
    image = cv.imread(image_path)
    detector.draw_bounding_box_and_save(image.copy(), sorted_boxes, output_image_path)
