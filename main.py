import argparse
import logging
import os
import sys
from datetime import datetime
from src.japanese_ocr.pipeline import OCRPipeline
from src.japanese_ocr.config import settings

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ocr_{timestamp}.log")

    # formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file

def main():
    log_file = setup_logging()
    logging.info(f"Logging initialized. Writing logs to {log_file}")

    parser = argparse.ArgumentParser(description="Japanese Handwriting OCR (Robust Architecture)")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--det_model", help="Override detection model path")
    parser.add_argument("--rec_model", help="Override recognition model path")
    parser.add_argument("--output", help="Override output directory")
    
    args = parser.parse_args()
    
    # Update settings from CLI args if provided
    if args.det_model:
        settings.det_model_dir = args.det_model
    if args.rec_model:
        settings.rec_model_path = args.rec_model
    if args.output:
        settings.output_dir = args.output
        
    pipeline = OCRPipeline()
    result = pipeline.process_image(args.image_path)
    
    if result:
        print("\n" + "="*50)
        print(f"SUCCESS! Processed {len(result.segments)} text segments.")
        print("="*50)
        print(f"📄 Output JSON  : {os.path.abspath(result.output_json_path)}")
        print(f"🖼️  Output Image : {os.path.abspath(result.output_image_path)}")
        print(f"📝 Log File     : {os.path.abspath(log_file)}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("❌ Processing FAILED. Check logs for details.")
        print(f"📝 Log File     : {os.path.abspath(log_file)}")
        print("="*50 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
