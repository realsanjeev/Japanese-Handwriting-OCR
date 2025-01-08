import numpy as np
import cv2 as cv
from src.utils import show_image

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
    image_height, _ = image.shape
    scale_ratio = H / image_height
    
    # Resize image to the target height
    resized_image = cv.resize(image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv.INTER_AREA)
    
    # Pad the resized image to match the desired dimensions
    padded_image = np.pad(resized_image, ((0, 0), (0, W - resized_image.shape[1])), mode="constant", constant_values=255)
    
    # Save the padded image for inspection
    cv.imwrite(f'test/padded_{image_path.split("_")[-1][:-3]}.jpg', padded_image)

    # Reshape to match the network input shape
    input_image = padded_image[None, None, :, :]
    # show_image(padded_image, "resize image")
    
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
    print("Bounding box shape:", np.array(bounding_boxes).shape)

    for idx, coordinates in enumerate(bounding_boxes[0]):
        points = np.array(coordinates, dtype=np.int32)
        print("Processing bounding box points shape:", points.shape)

        x, y, w, h = cv.boundingRect(points)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv.fillPoly(mask, [points], 255)

        masked_image = cv.bitwise_and(image, image, mask=mask)

        offset = int(offset)
        cropped_image = masked_image[
            max(y - offset, 0): min(y + h + offset, image_height),
            max(x - offset, 0): min(x + w + offset, image_width)
        ]
        print("-------------------------")
        # show_image(image, "msaked image" )
        # show_image(cropped_image, "cropped_image")

        crop_path = f"{path_to_save}/crop_{idx}.jpg"
        cv.imwrite(crop_path, cropped_image)
        print(f"Cropped image saved at: {crop_path}")