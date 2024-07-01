import numpy as np
from openvino.runtime import Core
from itertools import groupby

class TextRecognition:
    def __init__(self, model_path):
        self.model_path = model_path
        self.core = Core(xml_config_file="/home/sanjeev/Desktop/test/Japanese-Handwritten-OCR/model/handwritten-japanese-recognition-0001/FP16/handwritten-japanese-recognition-0001.xml")  # Initialize Core from openvino.runtime
        self.model = self.load_model()
        self.input_tensor, self.output_tensor = self.get_model_tensors()
        self.input_shape = self.input_tensor.shape

    def load_model(self):
        model = self.core.read_model(model=f"{self.model_path}.xml")
        compiled_model = self.core.compile_model(model, device_name="CPU")
        return compiled_model

    def get_model_tensors(self):
        input_tensor = next(iter(self.model.inputs))
        output_tensor = next(iter(self.model.outputs))
        return input_tensor, output_tensor

    def get_predictions_index(self, input_image):
        input_data = {self.input_tensor.name: input_image}  # Use name attribute
        results = self.model.infer(inputs=input_data)
        predictions = results[self.output_tensor.name]
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
