import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    det_model_dir: str = Field(
        default="model/det_model/ch_PP-OCRv4_det_infer.tar",
        description="Path to the detection model directory or tar file."
    )
    rec_model_path: str = Field(
        default="model/handwritten-japanese-recognition-0001/FP32/handwritten-japanese-recognition-0001",
        description="Path to the recognition model (without extension)."
    )
    charlist_path: str = Field(
        default="charlists/japanese_charlist.txt",
        description="Path to the character list file."
    )
    output_dir: str = Field(
        default="output",
        description="Directory to save output files."
    )
    
    # Model parameters
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    
    class Config:
        env_prefix = "OCR_"

# Global settings instance
settings = Settings()
