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


def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()



class Flatten(Module):
    """ Flat tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """ Convolution block without no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """ Global Norm-Aware Pooling block
    """
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
    """ Global Depthwise Convolution block
    """
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c,
                                     groups=in_c,
                                     kernel=(7, 7),
                                     stride=(1, 1),
                                     padding=(0, 0))
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
    """ SE block
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x
    
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        # Each convolution produces out_channels channels
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.conv3 = Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.conv = DepthwiseSeparableConvolution(in_channel=out_channels*3, kernels_per_layer=3, out_channel=out_channels)
        self.bn = BatchNorm2d(out_channels)
        self.act = PReLU(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1,x2,x3), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup


        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
    
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.
    
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.PReLU, gate_fn=hard_sigmoid, divisor=4, **_):
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

class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.PReLU(features)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.PReLU(d))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V


    
class DepthwiseSeparableConvolution(Module):
    def __init__(self, in_channel, kernels_per_layer, out_channel, stride=1,):
        super(DepthwiseSeparableConvolution,self).__init__()
        self.depthwise = Sequential(
            Conv2d(in_channel, in_channel * kernels_per_layer, kernel_size=3, padding=1, groups=in_channel, stride=stride, bias=False),
            BatchNorm2d(in_channel * kernels_per_layer),
            PReLU(in_channel * kernels_per_layer)
            )
        self.pointwise = Sequential(
            Conv2d(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False),
            BatchNorm2d(out_channel),
            )
    
    def forward(self, x): 
        out = self.depthwise(x) 
        out = self.pointwise(out) 
        return out


class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride, extra=False, multi=False,se=False, sk=False, kernel=3):
        super(BasicBlockIR, self).__init__()
        
        self.is_extra = extra
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False),BatchNorm2d(depth))

        # self.prelu = PReLU(depth)
        
        self.se = se
        self.sk = sk
        if self.is_extra:
            self.res_layer_1 = nn.Sequential(
                # depthwise
                BatchNorm2d(in_channel),
                Conv2d(in_channel, in_channel, kernel_size=kernel, padding=1, groups=in_channel, stride=stride, bias=False),
                BatchNorm2d(in_channel),
                PReLU(in_channel)
            )
            self.res_layer_2 = Sequential(
                # pointwise
                Conv2d(in_channel, depth, kernel_size=1, bias=False),
                BatchNorm2d(depth),
            )
        
            
        else:
            self.res_layer_1 = Sequential(
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (kernel, kernel), (1, 1), 1, bias=False),
                BatchNorm2d(depth),
                PReLU(depth),
            )
            self.res_layer_2 = Sequential(
                Conv2d(depth, depth, (kernel, kernel), stride, 1, bias=False),
                BatchNorm2d(depth)
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



class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride', 'extra','multi','se', 'sk', 'kernel'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2, extra=False,multi=False, se=False, sk=False, kernel=3):
    return [Bottleneck(in_channel, depth, stride,extra,multi,se, False, kernel )] +\
           [Bottleneck(depth, depth, 1,extra,False,False, sk, kernel) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=2,extra=True, se=False),
            get_block(in_channel=64, depth=128, num_units=2,extra=True, se=True),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=2,extra=False, se=True),
            get_block(in_channel=256, depth=512, num_units=2,extra=False, se=True),
        ]
    elif num_layers == 34:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3,extra=True, se=False, kernel=1),
            get_block(in_channel=64, depth=128, num_units=4,extra=True, se=True,kernel=1),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=6,extra=False, se=True),
            get_block(in_channel=256, depth=512, num_units=3,extra=False, se=True),
        ]
    elif num_layers == 50:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3,extra=True, se=False, kernel=3),
            get_block(in_channel=64, depth=128, num_units=4,extra=True, se=False,kernel=3),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=14,extra=False, se=False),
            get_block(in_channel=256, depth=512, num_units=3,extra=False, se=False),
        ]
    elif num_layers == 20:
        blocks1 = [
            get_block(in_channel=64, depth=128, num_units=5,extra=False, se=True, kernel=7, stride=4),
        ]
        blocks2 = [
            get_block(in_channel=128, depth=256, num_units=6,extra=False, se=True),
            get_block(in_channel=256, depth=512, num_units=3,extra=False, se=True),
        ]


    return [blocks1,blocks2]


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.PReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class ModifiedGDC(nn.Module):
    def __init__(self, image_size, in_chs, dropout, emb=512): 
        super(ModifiedGDC, self).__init__()
        if image_size % 32 == 0:
            self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=(image_size//32), groups=in_chs, bias=False)
        else:
            self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=(image_size//32 + 1), groups=in_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_chs, emb, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(emb)
    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten
        emb = self.bn2(x)
        return emb

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

class ConvBnActBuilder(namedtuple('ConvBnActBuilder', ['in_channel', 'out_channel', 'kernel_size', 'stride'])):
    "ConvBnActBuilder implementation"


class BackboneMod(Module):
    def __init__(self, input_size, num_layers):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(BackboneMod, self).__init__()
        assert input_size[0] in [112], \
            "input_size should be [112, 112]"
        assert num_layers in [18, 34,20, 50], \
            "num_layers should be 18, 34"
        # self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
        #                               BatchNorm2d(64), PReLU(64))

        self.input_layer = DepthwiseSeparableConvolution(in_channel=3, kernels_per_layer=3, out_channel=64)


        
        blocks = get_blocks(num_layers)
        unit_module = BasicBlockIR
        output_channel = 512


        self.output_layer = Sequential(BatchNorm2d(output_channel),
                                        Dropout(0.4), Flatten(),
                                        Linear(output_channel * 7 * 7, 512),
                                        BatchNorm1d(512, affine=False))

        modules = []
        last_ch = 0
        
        for block in blocks[0]:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride, bottleneck.extra,bottleneck.multi,bottleneck.se, bottleneck.sk, bottleneck.kernel))
                last_ch = bottleneck.depth
                
        

        
        for block in blocks[1]:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride, bottleneck.extra,bottleneck.multi,bottleneck.se,bottleneck.sk,bottleneck.kernel))
                last_ch = bottleneck.depth
        
        self.body = Sequential(*modules)
        
        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output,norm,x


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
