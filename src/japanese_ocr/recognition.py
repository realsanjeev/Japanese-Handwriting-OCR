import os
import numpy as np
from openvino import Core
from itertools import groupby
from typing import List
from .config import settings

class TextRecognizer:
    def __init__(self):
        self.files_checked = False
        self.core = Core()
        self.model = self._load_model()
        self.input_tensor, self.output_tensor = self._get_model_tensors()
        self.H, self.W = self._get_expected_hw()
        self.letters = self._load_charlist()

    def _load_model(self):
        model_xml = f"{settings.rec_model_path}.xml"
        if not os.path.exists(model_xml):
            raise FileNotFoundError(f"Model XML not found: {model_xml}")
            
        model = self.core.read_model(model=model_xml)
        return self.core.compile_model(model, device_name="CPU")

    def _get_model_tensors(self):
        input_tensor = next(iter(self.model.inputs))
        output_tensor = next(iter(self.model.outputs))
        return input_tensor, output_tensor
    
    def _get_expected_hw(self):
        _, _, H, W = self.input_tensor.shape
        return H, W

    def _load_charlist(self) -> str:
        if not os.path.exists(settings.charlist_path):
             raise FileNotFoundError(f"Charlist not found: {settings.charlist_path}")
             
        blank_char = "~"
        with open(settings.charlist_path, "r", encoding="utf-8") as f:
            letters = blank_char + "".join(line.strip() for line in f)
        return letters

    def predict(self, input_image: np.ndarray) -> str:
        """
        Runs inference on a single pre-processed image chunk.
        input_image: shape [1, 1, H, W]
        """
        predictions = self.model([input_image])[self.output_tensor]
        predictions = np.squeeze(predictions)
        predictions_index = np.argmax(predictions, axis=1)
        return self._decode(predictions_index)

    def _decode(self, predictions_index: np.ndarray) -> str:
        # Group duplicates (CTC decoding)
        output_text_indexes = [k for k, _ in groupby(predictions_index)]
        # Remove blanks (index 0 is usually blank in this specific model's logic/charlist)
        # Note: In the original code, it filtered out 0.
        valid_indexes = [i for i in output_text_indexes if i != 0]
        
        chars = [self.letters[i] for i in valid_indexes]
        return "".join(chars)
