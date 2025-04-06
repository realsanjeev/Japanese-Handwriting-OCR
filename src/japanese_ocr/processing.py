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
    Crops a detected text box using perspective transform to straighten it.
    Args:
        image: Source image.
        points: List of 4 points [(x,y), ...] from the detector.
        offset: Padding offset (not fully utilized in strict warp, but kept for signature).
    Returns:
        The straightforward, unwarped (deskewed) text image.
    """
    points = np.array(points, dtype=np.float32)
    
    # PaddleOCR output usually ordered: TL, TR, BR, BL
    # But let's robustly determine width/height
    
    # 1. Order points (standard assumption for Paddle is good, but let's be safe if needed)
    #    (tl, tr, br, bl) = points
    tl, tr, br, bl = points
    
    # 2. Compute width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # 3. Compute height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 4. Construct destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    # 5. Compute the Perspective Transform Matrix and Warp
    M = cv.getPerspectiveTransform(points, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped
