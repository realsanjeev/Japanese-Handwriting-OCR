import numpy as np
from openvino.runtime import Core
from itertools import groupby

class TextRecognition:
    def __init__(self, model_path):
        self.model_path = model_path
        self.core = Core()  # Initialize Core from openvino.runtime
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
        infer_request = self.model.create_infer_request()
        results = infer_request.infer(inputs=input_data)
        predictions = results[self.output_tensor.get_any_name()]
        predictions = np.squeeze(predictions)
        predictions_index = np.argmax(predictions, axis=1)
        return predictions_index

    @staticmethod
    def prepare_charlist():
        blank_char = "~"
        with open('charlists/japanese_charlist.txt', "r", encoding="utf-8") as charlist:
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
