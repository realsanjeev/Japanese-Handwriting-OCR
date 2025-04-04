import cv2 as cv
import numpy as np
from typing import List, Tuple

def apply_threshold(image: np.ndarray) -> np.ndarray:
    """
    Applies Otsu's thresholding to a grayscale image.
    Args:
        image: Input image (grayscale or color).
    Returns:
        Thresholded binary image.
    """
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    _, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresholded

def resize_image(H: int, W: int, image: np.ndarray) -> np.ndarray:
    """
    Resizes an image to the given height and width while maintaining aspect ratio.
    Pads with white if necessary.
    """
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
    image_height, image_width = image.shape
    
    scale = min(H / image_height, W / image_width)
    new_height = int(image_height * scale)
    new_width = int(image_width * scale)
    
    resized = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    pad_h = H - new_height
    pad_w = W - new_width
    
    padded = np.pad(resized, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
    
    # Add batch and channel dimensions for OpenVINO [1, 1, H, W]
    return padded[None, None, :, :]

def crop_box(image: np.ndarray, points: List[Tuple[float, float]], offset: int = 0) -> np.ndarray:
    """
    Crops a single bounding box from the image.
    """
    image_height, image_width = image.shape[:2]
    np_points = np.array(points, dtype=np.int32)
    
    x, y, w, h = cv.boundingRect(np_points)
    
    # Create mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv.fillPoly(mask, [np_points], 255)
    
    # Mask image (white background)
    masked = np.full_like(image, 255)
    cv.copyTo(src=image, mask=mask, dst=masked)
    
    # Crop
    y_min = max(y - offset, 0)
    y_max = min(y + h + offset, image_height)
    x_min = max(x - offset, 0)
    x_max = min(x + w + offset, image_width)
    
    return masked[y_min:y_max, x_min:x_max]
