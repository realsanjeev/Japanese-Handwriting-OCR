import cv2 as cv
import os
import sys
from src.detect_text import TextDetection
from src.recognize_text import TextRecognition
from src.pre_process import apply_threshold, resize_image, crop_image
from src.utils import save_to_json, make_folder, del_folder, list_file_paths_in_folder, show_image


def main():
    INPUT_IMAGE_PATH = sys.argv[1]
    FILE_NAME = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    # Make a temporary folder to store pre-processed images
    TEMP_FOLDER = os.path.splitext(INPUT_IMAGE_PATH)[0]
    # Make a folder to store cropped images and threshold images
    CROPPED_IMAGES_FOLDER = f"{TEMP_FOLDER}/cropped_images"
    THRESHOLDED_IMAGES_FOLDER = f"{TEMP_FOLDER}/thresholded_images"
    OUTPUT_PATH = "output"

    # initialize the detection model
    text_detection = TextDetection(det_model_dir='model/det_model/ch_ppocr_server_v2.0_det_infer/')
    text_recognition = TextRecognition('model/handwritten-japanese-recognition-0001/FP32/handwritten-japanese-recognition-0001')
    image = cv.imread(INPUT_IMAGE_PATH)
    # show_image(image, "Input Image")
    bounding_boxes = text_detection.detect_text_coordinates(INPUT_IMAGE_PATH)
    sorted_bounding_boxes = text_detection.sort_bounding_boxes(bounding_boxes)

    make_folder(CROPPED_IMAGES_FOLDER)
    crop_image(image=image, bounding_boxes=sorted_bounding_boxes, path_to_save=CROPPED_IMAGES_FOLDER)
    
    # Store text detected images to 'output' folder
    make_folder(OUTPUT_PATH)   
    text_detection.draw_bounding_box_and_save(image, 
                                              sorted_bounding_boxes, 
                                              output_image_path=f'{OUTPUT_PATH}/{FILE_NAME}.jpg')

    # Make a folder to store thresholded images
    make_folder(THRESHOLDED_IMAGES_FOLDER)

    letters = text_recognition.prepare_charlist()
    extracted_text = {}

    cropped_image_paths = list_file_paths_in_folder(CROPPED_IMAGES_FOLDER)
    for idx, cropped_image_path in enumerate(cropped_image_paths):
        # for easier debugging make index of threshold and cropped image same
        index = cropped_image_path.split('/')[-1].split('_')[1].split('.')[0]
        # print(index)
        apply_threshold(cropped_image_path, save_path=f'{THRESHOLDED_IMAGES_FOLDER}/threshold_{index}.jpg')

        input_image = resize_image(text_recognition.H, text_recognition.W, f'{THRESHOLDED_IMAGES_FOLDER}/threshold_{index}.jpg')
        predictions_index = text_recognition.get_predictions_index(input_image)
        output_text = text_recognition.get_text_from_predictions_index(predictions_index, letters)
        extracted_text[int(index)+1] = output_text

    sorted_dict = {key: extracted_text[key] for key in sorted(extracted_text.keys(), key=int, reverse=False)}
    save_to_json(sorted_dict, save_path=f'{OUTPUT_PATH}/{FILE_NAME}.json')

    # Delete temporary folder created earlier
    del_folder(TEMP_FOLDER)

if __name__ == "__main__":
    main()
