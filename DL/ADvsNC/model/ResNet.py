import torch.nn as nn

class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""

    def __init__(self, num_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 2, 2))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

        # self.attention1 = SELayer(64 * block.expansion)
        # self.attention2 = SELayer(128 * block.expansion)
        # self.attention3 = SELayer(256 * block.expansion)
        self.attention4 = SELayer(512 * block.expansion)

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1)):
        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

        # ------------------------------------------------------------------

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # x = self.attention1(x)

        x = self.layer2(x)
        # x = self.attention2(x)

        x = self.layer3(x)
        # x = self.attention3(x)

        x = self.layer4(x)
        x = self.attention4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ResNet3D
def resnet10_3d(**kwargs):
    return ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet18_3d(**kwargs):
    return ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_3d(**kwargs):
    return ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_3d(**kwargs):
    return ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_3d(**kwargs):
    return ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_3d(**kwargs):
    return ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet200_3d(**kwargs):
    return ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
