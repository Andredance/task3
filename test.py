import numpy as np

from digit_classifier import DigitClassifier


def test_algorithms():
    rand_img = np.random.randint(0, 256, (29, 28, 1))
    cls = DigitClassifier(algorithm="cnn")
    assert isinstance(cls.predict(rand_img), int)

    cls = DigitClassifier(algorithm="rf")
    assert isinstance(cls.predict(rand_img), int)

    cls = DigitClassifier(algorithm="rand")
    assert isinstance(cls.predict(rand_img), int)


if __name__ == "__main__":
    test_algorithms()
