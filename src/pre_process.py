import cv2 as cv

class PreProcessing:

    @staticmethod
    def apply_threshold(image_path, save_path):
        # Read the image in grayscale
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        # Apply Otsu's thresholding
        _, thresholded_image = cv.threshold(
            image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Save the thresholded image
        cv.imwrite(save_path, thresholded_image)

    @staticmethod
    def resize_image(H, W, image_path):
        # Read the image in grayscale
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image_height, _ = image.shape

        # Calculate the scale ratio to resize image to the target height
        scale_ratio_height = H / image_height
        
        # Resize image to the target height
        resized_image = cv.resize(
            image, None, fx=scale_ratio_height, fy=scale_ratio_height, interpolation=cv.INTER_AREA)
        cv.imwrite(f'test/resizedpadded_{image_path.split("_")[-1][:-3]}.jpg', resized_image)
        resized_image_width = resized_image.shape[1]
        scale_ratio_width = W / resized_image_width
        print("*"*100, scale_ratio_width)

        if scale_ratio_width < 1:
            resized_image = cv.resize(
                resized_image, None, fx=scale_ratio_width, fy=scale_ratio_width, interpolation=cv.INTER_AREA)
            height_difference = H - resized_image.shape[0]

            # Determine padding
            pad_top = height_difference // 2
            pad_bottom = height_difference - pad_top
            padded_image = cv.copyMakeBorder(
                resized_image, top=pad_top, bottom=pad_bottom, left=0, right=0, borderType=cv.BORDER_CONSTANT, value=255)
        else:
            padded_image = cv.copyMakeBorder(
                resized_image, top=0, bottom=0, left=0, right=(W - resized_image.shape[1]), borderType=cv.BORDER_CONSTANT, value=255)

        width, height = padded_image.shape
        print(f"Padded Image: {image_path}, Width: {width}, Height: {height}")
        cv.imwrite(f'test/padded_{image_path.split("_")[-1][:-3]}.jpg', padded_image)

        # Reshape to match the network input shape
        input_image = padded_image[None, None, :, :]
        
        return input_image
