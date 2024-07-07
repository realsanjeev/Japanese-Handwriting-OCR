import os
import shutil
import json
import cv2 as cv
from typing import Any, List

def save_to_json(data: Any, save_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (Any): The data to be saved to JSON.
        save_path (str): The file path where the JSON will be saved.
    """
    with open(save_path, 'wb') as fp:
        fp.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    print("Output text saved to JSON.")

def show_image(image: Any, window_name: str) -> None:
    """
    Display an image in a window.

    Args:
        image (Any): The image to be displayed.
        window_name (str): The name of the window where the image will be displayed.
    """
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def make_folder(folder_path: str) -> None:
    """
    Create a new folder if it doesn't already exist.

    Args:
        folder_path (str): The path of the folder to be created.
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

def del_folder(folder_path: str) -> None:
    """
    Delete a folder and all of its contents.

    Args:
        folder_path (str): The path of the folder to be deleted.
    """
    shutil.rmtree(folder_path)

def list_file_paths_in_folder(folder_path: str) -> List[str]:
    """
    List all file paths in a folder, sorted alphabetically.

    Args:
        folder_path (str): The path of the folder to list files from.

    Returns:
        List[str]: A list of file paths in the specified folder.
    """
    file_paths = []
    try:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                file_paths.append(file_path)
        file_paths.sort()
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return file_paths
