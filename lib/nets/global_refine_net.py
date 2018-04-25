import torch.nn as nn
import torch
import layer_utils
import math

class globalNet(nn.Module):

    def __init__(self, input_sizes, output_shape, num_points):
        super(globalNet, self).__init__()

        self.layer1_1 = self._make_layer1(input_sizes[0])
        self.layer1_2 = self._make_layer2()
        self.layer1_3 = self._make_layer3(output_shape, num_points)

        self.layer2_1 = self._make_layer1(input_sizes[1])
        self.layer2_2 = self._make_layer2()
        self.layer2_3 = self._make_layer3(output_shape, num_points)

        self.layer3_1 = self._make_layer1(input_sizes[2])
        self.layer3_2 = self._make_layer2()
        self.layer3_3 = self._make_layer3(output_shape, num_points)

        self.layer4_1 = self._make_layer1(input_sizes[3])
        self.layer4_3 = self._make_layer3(output_shape, num_points)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer1(self, input_size):

        layers = []

        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_layer2(self):

        layers = []

        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=True))

        return nn.Sequential(*layers)

    def _make_layer3(self, output_shape, num_points):

        layers = []

        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_points,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_points))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear'))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1_1 = self.layer1_1(x[0])
        x1_2 = self.layer1_2(x1_1)
        x1_3 = self.layer1_3(x1_1)

        x2_1 = self.layer2_1(x[1]) + x1_2
        x2_2 = self.layer2_2(x2_1)
        x2_3 = self.layer2_3(x2_1)

        x3_1 = self.layer3_1(x[2]) + x2_2
        x3_2 = self.layer3_2(x3_1)
        x3_3 = self.layer3_3(x3_1)

        x4_1 = self.layer4_1(x[3]) + x3_2
        x4_3 = self.layer4_3(x4_1)

        return [x4_1, x3_1, x2_1, x1_1], [x4_3, x3_3, x2_3, x1_3]

class refineNet(nn.Module):

    def __init__(self, input_size, out_shape, num_points):
        super(refineNet, self).__init__()
        self.layer1 = self._make_layer1(input_size, 0, out_shape)
        self.layer2 = self._make_layer1(input_size, 1, out_shape)
        self.layer3 = self._make_layer1(input_size, 2, out_shape)
        self.layer4 = self._make_layer1(input_size, 3, out_shape)

        self.final_branch = self._make_layer2(1024, num_points)

    def _make_layer1(self, input_size, num, output_shape):

        layers = []

        for i in range(num):
            layers.append(layer_utils.Bottleneck(input_size, 128))

        layers.append(nn.Upsample(size=output_shape, mode='bilinear'))

        return nn.Sequential(*layers)

    def _make_layer2(self, input_size, num_points):

        layers = []

        layers.append(layer_utils.Bottleneck(input_size, 128))
        layers.append(nn.Conv2d(256, num_points,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_points))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.layer1(x[0])
        x2 = self.layer2(x[1])
        x3 = self.layer3(x[2])
        x4 = self.layer4(x[3])

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.final_branch(out)

        return out
