import numpy as np

from sklearn.ensemble import RandomForestClassifier as RFClassifier

from classifier import DigitClassificationInterface


class RandomForestClassifier(DigitClassificationInterface):
    def __init__(self):
        self.rf_classifier = RFClassifier()
        # mock for train process
        self.rf_classifier.fit(np.random.randint(0, 256, (10, 784)), np.random.randint(0, 11, 10))

    def preprocess(self, image: np.ndarray):
        return image.reshape(1, -1)

    def raw_predict(self, model_input):
        return self.rf_classifier.predict(model_input)

    def postprocess(self, model_output: np.ndarray) -> int:
        return model_output[0].item()
