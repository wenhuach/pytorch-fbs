import torch.nn as nn
import torch.nn.functional as F
import torch


def conv2d_same_padding(input, weight, bias=None, stride=[1, 1], dilation=[1, 1], groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class TorchGraph(object):
    def __init__(self):
        self._graph = {}

    def add_tensor_list(self, name):
        self._graph[name] = []

    def append_tensor(self, name, val):
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        self._graph[name].clear()

    def get_tensor_list(self, name):
        return self._graph[name]

    def set_global_var(self, name, val):
        self._graph[name] = val

    def get_global_var(self, name):
        return self._graph[name] 


_Graph = TorchGraph()
_Graph.add_tensor_list('gate_values')
_Graph.set_global_var('ratio', 1.0)


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gated=True):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(out_channels, affine=False)

        if gated:
            self.gate = nn.Linear(in_channels, out_channels)
            self.beta = nn.Parameter(torch.Tensor(out_channels))
            self.ratio = 1.0

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

    def gated_forward(self, x):
        upsampled = F.avg_pool2d(x, x.shape[2])
        upsampled = upsampled.view(x.shape[0], x.shape[1])
        gates = F.relu(self.gate(upsampled))
        if self.training:
            _Graph.append_tensor('gate_values', gates)

        beta = self.beta.repeat(x.shape[0], 1)
        self.ratio = _Graph.get_global_var('ratio')

        if self.ratio < 1:
            inactive_channels = self.conv.out_channels - round(self.conv.out_channels * self.ratio)
            inactive_idx = (-gates).topk(inactive_channels, 1)[1]
            gates.scatter_(1, inactive_idx, 0)  # set inactive channels as zeros
            beta.scatter_(1, inactive_idx, 0)

        x = self.conv(x)
        x = self.bn(x)
        x = gates.unsqueeze(2).unsqueeze(3) * x
        x = x + beta.unsqueeze(2).unsqueeze(3)
        x = F.relu(x)

        return x


class CifarNet(nn.Module):
    def __init__(self, gated=True):
        super(CifarNet, self).__init__()
        self.gconv0 = GatedConv(3, 64, padding=0, gated=gated)
        self.gconv1 = GatedConv(64, 64, gated=gated)
        self.gconv2 = GatedConv(64, 128, stride=2, gated=gated)
        self.gconv3 = GatedConv(128, 128, gated=gated)
        self.drop3 = nn.Dropout2d()
        self.gconv4 = GatedConv(128, 128, gated=gated)
        self.gconv5 = GatedConv(128, 192, stride=2, gated=gated)
        self.gconv6 = GatedConv(192, 192, gated=gated)
        self.drop6 = nn.Dropout2d()
        self.gconv7 = GatedConv(192, 192, gated=gated)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.gconv0(x)
        x = self.gconv1(x)
        x = self.gconv2(x)
        x = self.gconv3(x)
        x = self.drop3(x)
        x = self.gconv4(x)
        x = self.gconv5(x)
        x = self.gconv6(x)
        x = self.drop6(x)
        x = self.gconv7(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)

        return  x

    def gated_forward(self, x):
        x = self.gconv0.gated_forward(x)
        x = self.gconv1.gated_forward(x)
        x = self.gconv2.gated_forward(x)
        x = self.gconv3.gated_forward(x)
        x = self.drop3(x)
        x = self.gconv4.gated_forward(x)
        x = self.gconv5.gated_forward(x)
        x = self.gconv6.gated_forward(x)
        x = self.drop6(x)
        x = self.gconv7.gated_forward(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)

        return  x
