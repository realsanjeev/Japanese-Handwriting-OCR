import numpy as np
import os
from openvino import Core
from itertools import groupby

class TextRecognition:
    def __init__(self, model_path):
        self.model_path = model_path
        self.core = Core()
        self.model = self.load_model()
        self.input_tensor, self.output_tensor = self.get_model_tensors()
        self.H, self.W = self.get_expected_height_and_weight()

    def load_model(self):
        model = self.core.read_model(model=f"{self.model_path}.xml")
        compiled_model = self.core.compile_model(model, device_name="CPU")
        return compiled_model

    def get_model_tensors(self):
        input_tensor = next(iter(self.model.inputs))
        output_tensor = next(iter(self.model.outputs))
        return input_tensor, output_tensor
    
    def get_expected_height_and_weight(self):
        _, _, H, W = self.input_tensor.shape
        return H, W

    def get_predictions_index(self, input_image):
        input_data = {self.input_tensor.get_any_name(): input_image}  # Use get_any_name() for tensor name
        # infer_request = self.model.create_infer_request()
        # results = infer_request.infer(inputs=input_data)
        # predictions = results[self.output_tensor.get_any_name()]
        predictions = self.model([input_image])[self.output_tensor]
        predictions = np.squeeze(predictions)
        predictions_index = np.argmax(predictions, axis=1)
        return predictions_index

    @staticmethod
    def prepare_charlist(charlist_path='charlists/japanese_charlist.txt'):
        blank_char = "~"
        if not os.path.exists(charlist_path):
             # Try to find it relative to the current script if not found directly
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
            charlist_path = os.path.join(project_root, charlist_path)
            
        if not os.path.exists(charlist_path):
            raise FileNotFoundError(f"Charlist file not found at: {charlist_path}")

        with open(charlist_path, "r", encoding="utf-8") as charlist:
            letters = blank_char + "".join(line.strip() for line in charlist)
        return letters

    @staticmethod
    def get_text_from_predictions_index(predictions_index, letters):
        output_text_indexes = list(groupby(predictions_index))
        output_text_indexes = np.array([key for key, _ in output_text_indexes])
        output_text_indexes = output_text_indexes[output_text_indexes != 0]
        output_char = [letters[letter_index] for letter_index in output_text_indexes]
        output_text = "".join(output_char)
        return output_text
