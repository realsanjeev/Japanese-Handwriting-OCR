import os
import numpy as np
import cv2 as cv
from src.utils import show_image, make_folder

def apply_threshold(image_path, save_path: str):
    """
    Applies Otsu's thresholding to an image and saves the thresholded image to the given path.

    Args:
        image_path (str): Path to the input image.
        save_path (str): Path to save the thresholded image.
    """
    # Read the image in grayscale
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Apply Otsu's thresholding
    _, thresholded_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    try:
        # Save the thresholded image
        cv.imwrite(save_path, thresholded_image)
    except Exception as e:
        print(f"Error saving image at {save_path}: {e}")
        print(f"Image type: {type(thresholded_image)}, Input image: {image_path}")


def resize_image(H, W, image_path: str):
    """
    Resizes an image to the given height and width while maintaining aspect ratio.
    The image is padded if necessary to match the target dimensions.

    Args:
        H (int): Target height.
        W (int): Target width.
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Resized and padded image suitable for input to a network.
    """
    # Read the image in grayscale
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Calculate the scale ratio to resize image to the target height
    image_height, image_width = image.shape
    
    # Scale to fit into H x W box while maintaining aspect ratio
    scale = min(H / image_height, W / image_width)
    
    new_height = int(image_height * scale)
    new_width = int(image_width * scale)
    
    # Resize image
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    # Pad the resized image to match the desired dimensions (H, W)
    # Padding right and bottom to keep the image top-left aligned
    pad_h = H - new_height
    pad_w = W - new_width
    
    padded_image = np.pad(resized_image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
    
    # Reshape to match the network input shape
    input_image = padded_image[None, None, :, :]
    
    return input_image

def crop_image(image, bounding_boxes, path_to_save: str, offset: int=0):
    """
    Crops bounding boxes from the image and saves them as individual images.

    Args:
        image (np.ndarray): Input image (uint8).
        bounding_boxes (np.ndarray): Bounding boxes of text with shape 
            (batch_size, num_bounding_boxes, num_points_per_box, 2). 
            Each box contains 4 points in the order [lt, rt, rb, lb].
        path_to_save (str): Path where cropped images will be saved.
        offset (int): Addtional boundry in detected character. 
            May lower accuracy of character recognization if image is used for OCR here
    """
    image_height, image_width = image.shape[:2]

    for idx, coordinates in enumerate(bounding_boxes[0]):
        points = np.array(coordinates, dtype=np.int32)

        x, y, w, h = cv.boundingRect(points)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv.fillPoly(mask, [points], 255)

        # Use 255 (white) for background instead of 0 (black)
        masked_image = np.full_like(image, 255)
        cv.copyTo(src=image, mask=mask, dst=masked_image)

        offset = int(offset)
        cropped_image = masked_image[
            max(y - offset, 0): min(y + h + offset, image_height),
            max(x - offset, 0): min(x + w + offset, image_width)
        ]

        crop_path = f"{path_to_save}/crop_{idx}.jpg"
        cv.imwrite(crop_path, cropped_image)