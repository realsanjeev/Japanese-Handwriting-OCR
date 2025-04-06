# Japanese Handwriting OCR

A robust, modular Optical Character Recognition (OCR) system for Japanese handwriting. This project utilizes:
-   **PaddleOCR** for both text detection and recognition.
-   **Pydantic** for robust configuration.
-   **In-Memory Processing** for high performance.
-   **Perspective Warping** for accurate text alignment.

![OCR Pipeline](./data/demos/architecture.png)

## 🚀 Features
-   **Architecture**: Modular design with separate detection, recognition, and pipeline logic.
-   **Performance**: Fast, in-memory image processing pipeline (no intermediate disk I/O).
-   **Robustness**: Strong error handling, logging, and type-safe configuration.
-   **Observability**: Detailed file logging and explicit console feedback.

## 🛠️ Prerequisites
-   Python 3.8+ and <3.13
-   Pip 24.0+

## 📥 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd Japanese-Handwriting-OCR
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### Basic Execution
Run OCR on an image using default settings:
```bash
python main.py data/image-03.jpeg
```

### Advanced Options
Customize execution with command-line arguments:
```bash
python main.py <image_path> \
    --output <dir>      # Custom output directory (default: output)
    --det_model <path>  # Path to custom detection model
    --rec_model <path>  # Path to custom recognition model
```

**Example**:
```bash
python main.py input.jpg --output my_results/ --det_model model/custom_det_model/
```

## 📂 Output
Upon successful execution, the system generates:
1.  **Annotated Image** (`.jpg`): The input image with detected bounding boxes and IDs drawn on it.
2.  **Result JSON** (`.json`): A structured file containing the text content for each detected segment.
3.  **Logs**: Detailed execution logs are saved in `logs/` directory.

Example Console Output:
```text
==================================================
SUCCESS! Processed 1 text segments.
==================================================
📄 Output JSON  : /path/to/project/output/image-03.json
🖼️  Output Image : /path/to/project/output/image-03.jpg
📝 Log File     : /path/to/project/logs/ocr_20251210_105653.log
==================================================
```

## 🏗️ Architecture

```text
src/japanese_ocr/
├── pipeline.py    # Orchestrates the Detection -> Recognition flow
├── config.py      # Pydantic-based settings management
├── domain.py      # Data models (BoundingBox, TextSegment)
├── detection.py   # Wrapper for PaddleOCR (Detection)
├── recognition.py # Wrapper for PaddleOCR (Recognition)
└── processing.py  # In-memory image operations (Perspective Warp, Crop)
```

## 🔗 References
- [PaddleOCR - Official Github Repo](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README.md)
- [Image Processing in OpenCV - OpenCV documentation](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)