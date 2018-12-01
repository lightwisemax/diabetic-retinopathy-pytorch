import torch
import math
import torch.nn as nn


class vgg(nn.Module):

    def __init__(self, num_classes=1):
        super(vgg, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
        )

        self.avg_pool = nn.AvgPool2d(8, 8)
        self.fc = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(512, num_classes, bias=False)
        )
        self._initialize_weights()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)


if __name__ == '__main__':
    tensor = torch.randn((1, 3, 128, 128))
    labels = torch.LongTensor([1, 0])
    model = vgg()

    params = list(model.parameters())
    print(model)
    output = model(tensor)
    import torch.nn.functional as F
    h_x = F.softmax(output, dim=1).data.squeeze()
    probs, idx = output.sort(0, True)
    print(idx[0])
    # criterion = nn.CrossEntropyLoss()
    # print(criterion(output, labels))


