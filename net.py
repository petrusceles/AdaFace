from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
from torch.nn import PReLU
import torch.nn.functional as F
import os


def build_model(model_name="ir_50"):
    if model_name == "ir_101":
        return IR_101(input_size=(112, 112))
    elif model_name == "ir_50":
        return IR_50(input_size=(112, 112))
    elif model_name == "ir_se_50":
        return IR_SE_50(input_size=(112, 112))
    elif model_name == "ir_34":
        return IR_34(input_size=(112, 112))
    elif model_name == "ir_18":
        return IR_18(input_size=(112, 112))
    else:
        raise ValueError("not a correct model name", model_name)


def initialize_weights(modules):
    """Weight initilize, conv2d and linear is initialized with kaiming_normal"""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """Flat tensor"""

    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """Convolution block without no-linear activation layer"""

    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(
            in_c, out_c, kernel, stride, padding, groups=groups, bias=False
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """Global Norm-Aware Pooling block"""

    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """Global Depthwise Convolution block"""

    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(
            in_c, in_c, groups=in_c, kernel=(7, 7), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """SE block"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class DepthwiseSeparableConvolution(Module):
    def __init__(
        self,
        in_channel,
        kernels_per_layer,
        out_channel,
        stride=1,
    ):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = Sequential(
            Conv2d(
                in_channel,
                in_channel * kernels_per_layer,
                kernel_size=3,
                padding=1,
                groups=in_channel,
                stride=stride,
                bias=False,
            ),
            BatchNorm2d(in_channel * kernels_per_layer),
        )
        self.pointwise = Sequential(
            Conv2d(
                in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False
            ),
            BatchNorm2d(out_channel),
            PReLU(out_channel),
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.PReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class BottleneckIR(Module):
    """BasicBlock with bottleneck for IRNet"""

    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIR(Module):
    """BasicBlock for IRNet"""

    def __init__(self, in_channel, depth, stride, extra=False, se=False, kernel=3):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )

        self.se = se
        self.is_extra = extra

        self.is_extra = extra
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )

        if self.is_extra:
            self.res_layer_1 = nn.Sequential(
                # depthwise
                BatchNorm2d(in_channel),
                Conv2d(
                    in_channel,
                    in_channel,
                    kernel_size=kernel,
                    padding=1,
                    groups=in_channel,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(in_channel),
                PReLU(in_channel),
            )
            self.res_layer_2 = Sequential(
                # pointwise
                Conv2d(in_channel, depth, kernel_size=1, bias=False),
                BatchNorm2d(depth),
            )
        else:
            self.res_layer_1 = Sequential(
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                BatchNorm2d(depth),
                PReLU(depth),
            )
            self.res_layer_2 = Sequential(
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth)
            )

        if self.se:
            self.se_layer = SqueezeExcite(in_chs=depth)

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer_1(x)
        res = self.res_layer_2(res)
        if self.se:
            res = self.se_layer(res)
        result = res + shortcut
        return result


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)


class Bottleneck(
    namedtuple("Block", ["in_channel", "depth", "stride", "extra", "se", "kernel"])
):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2, extra=False, se=False, kernel=3):

    return [Bottleneck(in_channel, depth, stride, extra, se, kernel)] + [
        Bottleneck(depth, depth, 1, extra, False, kernel) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=2, extra=True, se=False),
            get_block(in_channel=64, depth=128, num_units=2, extra=True, se=True),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=2, extra=False, se=True),
            get_block(in_channel=256, depth=512, num_units=2, extra=False, se=True),
        ]
    elif num_layers == 20:
        blocks1 = [
            get_block(
                in_channel=64,
                depth=128,
                num_units=7,
                extra=True,
                se=True,
                kernel=7,
                stride=4,
            ),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=6, extra=False, se=True),
            get_block(in_channel=256, depth=512, num_units=3, extra=False, se=True),
        ]
    elif num_layers == 34:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3, extra=True, se=False),
            get_block(in_channel=64, depth=128, num_units=4, extra=True, se=True),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=6, extra=False, se=True),
            get_block(in_channel=256, depth=512, num_units=3, extra=False, se=True),
        ]
    elif num_layers == 50:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3, extra=True, kernel=3),
            get_block(in_channel=64, depth=128, num_units=4, extra=True, kernel=3),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks1 = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
        ]
        blocks2 = [
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3),
        ]
    elif num_layers == 200:
        blocks1 = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
        ]
        blocks2 = [
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3),
        ]

    return [blocks1, blocks2]


class BackboneMod(Module):
    def __init__(self, input_size, num_layers, mode="ir"):
        """Args:
        input_size: input_size of BackboneMod
        num_layers: num_layers of BackboneMod
        mode: support ir or irse
        """
        super(BackboneMod, self).__init__()
        assert input_size[0] in [
            112,
            224,
        ], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [
            18,
            34,
            50,
            100,
            152,
            200,
        ], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        self.input_layer = DepthwiseSeparableConvolution(
            in_channel=3, kernels_per_layer=3, out_channel=64
        )
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == "ir":
                unit_module = BasicBlockIR
            elif mode == "ir_se":
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == "ir":
                unit_module = BottleneckIR
            elif mode == "ir_se":
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 7 * 7, 512),
                BatchNorm1d(512, affine=False),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False),
            )

        modules = []
        last_ch = 0
        for block in blocks[0]:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride,
                        bottleneck.extra,
                        bottleneck.se,
                    )
                )
                last_ch = bottleneck.depth

        for block in blocks[1]:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride,
                        bottleneck.extra,
                        bottleneck.se,
                    )
                )
                last_ch = bottleneck.depth
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):

        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm, x


def IR_18(input_size):
    """Constructs a ir-18 model."""
    model = BackboneMod(input_size, 18, "ir")

    return model


def IR_34(input_size):
    """Constructs a ir-34 model."""
    model = BackboneMod(input_size, 34, "ir")

    return model


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = BackboneMod(input_size, 50, "ir")

    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = BackboneMod(input_size, 100, "ir")

    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = BackboneMod(input_size, 152, "ir")

    return model


def IR_200(input_size):
    """Constructs a ir-200 model."""
    model = BackboneMod(input_size, 200, "ir")

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = BackboneMod(input_size, 50, "ir_se")

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = BackboneMod(input_size, 100, "ir_se")

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = BackboneMod(input_size, 152, "ir_se")

    return model


def IR_SE_200(input_size):
    """Constructs a ir_se-200 model."""
    model = BackboneMod(input_size, 200, "ir_se")

    return model
