from torch import nn
from nas_bib.utils.registre import register_class
import torch
import torch.nn.functional as F
import math

#IMPORTANT: pour chaque op, il faut rajouter: "@register_class(registry="operations")" et initlser ces paramtres self.out_channels pour que je puisse lui acceder apres

@register_class(registry="operations")
class sum_features_maps:
    """
    Additionne plusieurs feature maps de tailles et de nombres de canaux différents
    en utilisant une convolution 1x1 pour réduire le nombre de canaux au minimum et en interpolant la taille.
    """

    def __init__(self):
        pass

    def __call__(self, feature_maps):
        max_size = max(map(lambda x: x.size()[-2:], feature_maps))
        min_channels = min(map(lambda x: x.size(1), feature_maps))

        upsampled_maps = []
        for feature_map in feature_maps:
            if feature_map.size()[-2:] != max_size:
                upsampled_maps.append(F.interpolate(feature_map, size=max_size, mode='bilinear', align_corners=False))
            else:
                upsampled_maps.append(feature_map)

        conv_layers = []
        for feature_map in upsampled_maps:
            if feature_map.size(1) != min_channels:
                conv_layers.append(nn.Conv2d(feature_map.size(1), min_channels, kernel_size=1))
            else:
                conv_layers.append(nn.Identity())

        processed_maps = []
        for i, feature_map in enumerate(upsampled_maps):
            processed_maps.append(conv_layers[i](feature_map))

        return sum(processed_maps)

@register_class(registry="operations")
class concat_features_maps:
    """
    Concatène plusieurs feature maps de tailles différentes en ajoutant du padding constant.
    """

    def __init__(self):
        pass

    def __call__(self, feature_maps):
        max_size = max(map(lambda x: x.size()[-2:], feature_maps))
        padded_maps = []
        for feature_map in feature_maps:
            if feature_map.size()[-2:] != max_size:
                padded_maps.append(F.adaptive_avg_pool2d(feature_map, max_size))
            else:
                padded_maps.append(feature_map)
        return torch.cat(padded_maps, dim=1)

@register_class(registry="operations")
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        # Initialize attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Create convolution layers
        self.depthwise_conv = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels)
        self.pointwise_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

@register_class(registry="operations")
class DilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate=1, stride=1, padding=0, bias=True, groups=1):
        super(DilatedConv2d, self).__init__()
        # Initialize attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        # Create convolution layer
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, dilation=self.dilation_rate, groups=self.groups, bias=self.bias)

    def forward(self, x):
        return self.conv(x)

@register_class(registry="operations")
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        # Initialize attributes (same as SeparableConv2d)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Create convolution layers with potentially different bias settings
        self.depthwise_conv = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, groups=self.in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

@register_class(registry="operations")
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=2, stride=1):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        
        # Pointwise linear projection
        self.conv1 = nn.Conv2d(in_channels, expansion_ratio * in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion_ratio * in_channels)

        # Depthwise separable convolution
        self.conv2 = nn.Conv2d(expansion_ratio * in_channels, expansion_ratio * in_channels, kernel_size=3, stride=stride, padding=1, groups=expansion_ratio * in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion_ratio * in_channels)

        # Pointwise linear projection
        self.conv3 = nn.Conv2d(expansion_ratio * in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        # If channels are different, stride needs to be modified
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Forward through the bottleneck layer

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        shortcut = self.shortcut(x)

        return F.relu(out + shortcut)  # Apply ReLU after residual addition

@register_class(registry="operations")
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, affine=False):
        super(ConvBNReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

    def get_embedded_ops(self):
        return None

#-------------From NASLib ajusted

@register_class(registry="operations")
class DepthwiseConv(nn.Module):
    """
    Depthwise convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, affine=True):
        super(DepthwiseConv, self).__init__()
        #initalisation
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.affine = affine

        # if out_channels % in_channels != 0:
        #     # Ajuster out_channels pour qu'il soit divisible par in_channels
        #     self.out_channels = int(math.ceil(out_channels / in_channels)) * in_channels

        self.op = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

@register_class(registry="operations")
class SepConvDARTS(nn.Module):
    """
    Implementation of Separable convolution operation as
    in the DARTS paper, i.e. 2 sepconv directly after another.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, affine=True):
        super(SepConvDARTS, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.affine = affine

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.in_channels,
                bias=False,
            ),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channels, affine=self.affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                groups=self.in_channels,
                bias=False,
            ),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
        )

    def forward(self, x):
        return self.op(x)


@register_class(registry="operations")
class DilConvDARTS(nn.Module):
    """
    Implementation of a dilated separable convolution as
    used in the DARTS paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation = 1, affine=True):
        super(DilConvDARTS, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.affine = affine

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.in_channels,
                bias=False,
            ),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
        )

    def forward(self, x):
        return self.op(x)


@register_class(registry="operations")
class StemNASLib(nn.Module):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super(StemNASLib, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.seq = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        return self.seq(x)
    
@register_class(registry="operations")
class Zero(nn.Module):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, in_channels=None, out_channels=None, stride=(1, 1)):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super(Zero, self).__init__()
        
        if isinstance(stride, int):
            stride = (stride, stride)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        if self.in_channels == self.out_channels:
            if self.stride == (1, 1):
                return x.mul(0.0)
            else:
                stride_height, stride_width = self.stride
                return x[:, :, ::stride_height, ::stride_width].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1], shape[2], shape[3] = self.out_channels, (shape[2] + self.stride[0] - 1) // self.stride[0], (shape[3] + self.stride[1] - 1) // self.stride[1]
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

@register_class(registry="operations")
class MBConv(nn.Module):
  """
  MBConv block implementation for MobileNetV2 in PyTorch.

  Args:
    in_filters: Number of input filters.
    out_filters: Number of output filters.
    kernel_size: Size of the depthwise convolution kernel.
    strides: Stride of the convolutions (default: 1).
    expansion_ratio: Channel expansion ratio (default: 1).
    depth_multiplier: Depth multiplier to adjust filter counts (default: 1).
  """

  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_ratio=1, depth_multiplier=1):

    super(MBConv, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.expansion_ratio = expansion_ratio
    self.depth_multiplier = depth_multiplier 
    self.use_residual = in_channels == out_channels and stride == 1
    expand_channels = in_channels * expansion_ratio

    self.expand_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=expand_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False
    )
    self.expand_bn = nn.BatchNorm2d(expand_channels)
    self.expand_relu = nn.ReLU(inplace=True)
    self.depthwise_conv = nn.Conv2d(
        in_channels=expand_channels,
        out_channels=expand_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size//2,
        groups=expand_channels,
        bias=False
    )
    self.depthwise_bn = nn.BatchNorm2d(expand_channels)
    self.project_conv = nn.Conv2d(
        in_channels=expand_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False
    )
    self.project_bn = nn.BatchNorm2d(out_channels)
    if self.use_residual:
        self.shortcut = nn.Identity()
    else:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

  def forward(self, x):
    out = self.expand_relu(self.expand_bn(self.expand_conv(x)))
    out = self.depthwise_bn(self.depthwise_conv(out))
    out = self.project_bn(self.project_conv(out))

    if self.use_residual:
        out = out + self.shortcut(x)
    else:
        out = self.shortcut(x) + out

    return out
