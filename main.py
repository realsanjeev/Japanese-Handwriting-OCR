import cv2 as cv
import os
import argparse
import tempfile
from src.detect_text import TextDetection
from src.recognize_text import TextRecognition
from src.pre_process import apply_threshold, resize_image, crop_image
from src.utils import save_to_json, list_file_paths_in_folder

def main():
    parser = argparse.ArgumentParser(description="Japanese Handwriting OCR")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--det_model", default="model/det_model/ch_PP-OCRv4_det_infer.tar", help="Path to detection model directory")
    parser.add_argument("--rec_model", default="model/handwritten-japanese-recognition-0001/FP32/handwritten-japanese-recognition-0001", help="Path to recognition model")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    input_image_path = args.image_path
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        return

    file_name = os.path.splitext(os.path.basename(input_image_path))[0]
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Initialize models
    try:
        if not os.path.exists(args.det_model) and not os.path.exists(os.path.join(args.det_model, 'inference.pdmodel')):
             # detection model path might be a tar file or a dir. PaddleOCR handles it? 
             # The original code passed a tar path OR a dir.
             pass 

        text_detection = TextDetection(det_model_dir=args.det_model)
        text_recognition = TextRecognition(args.rec_model)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    image = cv.imread(input_image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    try:
        bounding_boxes = text_detection.detect_text_coordinates(input_image_path)
        sorted_bounding_boxes = text_detection.sort_bounding_boxes(bounding_boxes)
    except Exception as e:
        print(f"Error during text detection: {e}")
        return

    # Use a temporary directory for intermediate steps
    with tempfile.TemporaryDirectory() as temp_dir:
        cropped_images_folder = os.path.join(temp_dir, "cropped_images")
        thresholded_images_folder = os.path.join(temp_dir, "thresholded_images")
        
        os.makedirs(cropped_images_folder, exist_ok=True)
        os.makedirs(thresholded_images_folder, exist_ok=True)
        
        # Save detection result
        output_image_path = os.path.join(args.output, f'{file_name}.jpg')
        text_detection.draw_bounding_box_and_save(image.copy(), sorted_bounding_boxes, output_image_path)
        
        crop_image(image=image, bounding_boxes=sorted_bounding_boxes, path_to_save=cropped_images_folder)
        
        letters = text_recognition.prepare_charlist()
        extracted_text = {}

        cropped_image_paths = list_file_paths_in_folder(cropped_images_folder)
        
        for cropped_image_path in cropped_image_paths:
            # Filename format: crop_{idx}.jpg
            try:
                base = os.path.basename(cropped_image_path)
                index = base.split('_')[1].split('.')[0]
                
                threshold_path = os.path.join(thresholded_images_folder, f'threshold_{index}.jpg')
                apply_threshold(cropped_image_path, save_path=threshold_path)

                input_image = resize_image(text_recognition.H, text_recognition.W, threshold_path)
                predictions_index = text_recognition.get_predictions_index(input_image)
                output_text = text_recognition.get_text_from_predictions_index(predictions_index, letters)
                extracted_text[int(index) + 1] = output_text
            except Exception as e:
                print(f"Error processing text segment {cropped_image_path}: {e}")
                continue

        sorted_dict = {key: extracted_text[key] for key in sorted(extracted_text.keys(), key=int, reverse=False)}
        save_to_json(sorted_dict, save_path=os.path.join(args.output, f'{file_name}.json'))
        print(f"Processing complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
