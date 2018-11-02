from torch import nn
import torch


class Locator(nn.Module):
    """
    Locator: a model architecture consisting of an unet and following classifier
    """
    def __init__(self, aer, classifier):
        super(Locator, self).__init__()
        self.aer = aer
        self.classifier = classifier

    def forward(self, x):
        code = self.aer(x)
        diff = code - x
        diff = torch.clamp(diff, min=0.0, max=1.0)
        outputs = self.classifier(diff)
        return code, outputs

if __name__ == '__main__':
    pass