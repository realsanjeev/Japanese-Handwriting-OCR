import os
from pydantic_settings import BaseSettings
from pydantic import Field

from typing import Optional

class Settings(BaseSettings):
    det_model_dir: str = Field(
        default="model/det_model/ch_PP-OCRv4_det_infer.tar",
        description="Path to the detection model directory or tar file."
    )
    rec_language: str = Field(
        default="japan",
        description="Language for recognition model (default: japan)."
    )
    rec_model_dir: Optional[str] = Field(
        default=None,
        description="Optional path to custom recognition model dir."
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
