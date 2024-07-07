import cv2 as cv
import os
import sys
from src.detect_text import TextDetection
from src.pre_process import PreProcessing
from src.crop_image import CropImage
from src.recognize_text import TextRecognition
from src.utils import save_to_json, make_folder, del_folder, list_file_paths_in_folder, show_image


def main():
    INPUT_IMAGE_PATH = sys.argv[1]
    FILE_NAME = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    text_detection = TextDetection(det_model_dir='model/det_model/ch_ppocr_server_v2.0_det_infer/')
    text_recognition = TextRecognition('model/handwritten-japanese-recognition-0001/FP32/handwritten-japanese-recognition-0001')
    image = cv.imread(INPUT_IMAGE_PATH)
    show_image(image, "Input Image")
    bounding_boxes = text_detection.detect_text_coordinates(INPUT_IMAGE_PATH)
    # print("Bounding Boxes: ", bounding_boxes)
    sorted_bounding_boxes = text_detection.sort_bounding_boxes(bounding_boxes)
    # print("Sorted Bounding Boxes: ", sorted_bounding_boxes)

    # Make a temporary folder to store pre-processed images
    temp_folder = os.path.splitext(INPUT_IMAGE_PATH)[0]

    # Make a folder to store cropped images
    cropped_images_folder = f"{temp_folder}/cropped_images"
    make_folder(cropped_images_folder)

    CropImage.crop_image(image=image, bounding_boxes=sorted_bounding_boxes, path_to_save=cropped_images_folder)
    
    # Store text detected images to 'output' folder
    output_path = "output"
    make_folder(output_path)   
    text_detection.draw_bounding_box_and_save(image, sorted_bounding_boxes, output_image_path=f'{output_path}/{FILE_NAME}.jpg')

    # Make a folder to store thresholded images
    thresholded_images_folder = f"{temp_folder}/thresholded_images"
    make_folder(thresholded_images_folder)

    letters = text_recognition.prepare_charlist()
    extracted_text = {}

    cropped_image_paths = list_file_paths_in_folder(cropped_images_folder)
    for idx, cropped_image_path in enumerate(cropped_image_paths):
        # for easier debugging make index of threshold and cropped image same
        index = cropped_image_path.split('/')[-1].split('_')[1].split('.')[0]
        # print(index)
        PreProcessing.apply_threshold(cropped_image_path, save_path=f'{thresholded_images_folder}/threshold_{index}.jpg')

        input_image = PreProcessing.resize_image(text_recognition.H, text_recognition.W, f'{thresholded_images_folder}/threshold_{index}.jpg')
        predictions_index = text_recognition.get_predictions_index(input_image)
        output_text = text_recognition.get_text_from_predictions_index(predictions_index, letters)
        extracted_text[index] = output_text

    sorted_dict = {key: extracted_text[key] for key in sorted(extracted_text.keys(), key=int, reverse=True)}
    save_to_json(sorted_dict, save_path=f'{output_path}/{FILE_NAME}.json')

    # Delete temporary folder created earlier
    del_folder(temp_folder)

if __name__ == "__main__":
    main()
