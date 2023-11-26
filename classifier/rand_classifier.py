import numpy as np

from classifier.base import DigitClassificationInterface


class RandomClassifier(DigitClassificationInterface):
    def preprocess(self, image: np.ndarray):
        return image[9: 19, 9: 19, :]

    def raw_predict(self, model_input: np.ndarray):
        return np.random.randint(0, 11, 1)

    def postprocess(self, model_output: np.ndarray) -> int:
        return model_output[0].item()
