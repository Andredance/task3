import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.base import DigitClassificationInterface


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CnnClassifier(DigitClassificationInterface):
    def __init__(self):
        self.model = Net()

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32)
        image = np.rollaxis(image, 2, 0)
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image)

    def raw_predict(self, model_input: torch.Tensor):
        return self.model(model_input)

    def postprocess(self, model_output: torch.Tensor) -> int:
        pred = model_output.argmax(dim=1)
        return pred.item()
