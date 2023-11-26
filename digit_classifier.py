from classifier import DigitClassificationInterface, CnnClassifier, RandomForestClassifier, RandomClassifier


class DigitClassifier:
    supported_algorithms = {"cnn", "rf", "rand"}
    _algorithms_match = {
        "cnn": CnnClassifier,
        "rf": RandomForestClassifier,
        "rand": RandomClassifier
    }

    def __init__(self, algorithm="cnn"):
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm {algorithm}!")

        self.model: DigitClassificationInterface = self._algorithms_match[algorithm]()

    def train(self):
        raise NotImplementedError()

    def predict(self, image):
        return self.model.predict(image)
