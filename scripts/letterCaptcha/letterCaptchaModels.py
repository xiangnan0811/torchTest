from torch import nn
from torchvision import models


class LetterCaptchaModel(nn.Module):
    def __init__(self):
        super(LetterCaptchaModel, self).__init__()
        self.resnet18 = models.resnet18(num_classes=4*8)

    def forward(self, x):
        x = self.resnet18(x)
        return x