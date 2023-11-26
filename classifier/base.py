import numpy as np

from abc import ABC, abstractmethod


class DigitClassificationInterface(ABC):
    @abstractmethod
    def preprocess(self, image: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def raw_predict(self, model_input):
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, model_output) -> int:
        raise NotImplementedError()

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28, 1):
            raise ValueError(f"Wrong image shape: {image.shape}. Expected (28, 28, 1)")

        model_input = self.preprocess(image)
        model_result = self.raw_predict(model_input)
        return self.postprocess(model_result)

