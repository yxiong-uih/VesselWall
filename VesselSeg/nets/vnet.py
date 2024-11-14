import torch.nn as nn
import torch
import numpy as np


class InputBlock(nn.Module):
    """ input block of vb-net """

    def __init__(self, in_channels, out_channels, bias = True):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class ConvBnRelu3(nn.Module):
    """ classic combination: conv + batch normalization [+ relu] """

    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True,bias=True):
        super(ConvBnRelu3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1,bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.bn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class BottConvBnRelu3(nn.Module):
    """Bottle neck structure"""

    def __init__(self, channels, ratio, do_act=True,bias = True):
        super(BottConvBnRelu3, self).__init__()
        self.conv1 = ConvBnRelu3(channels, channels//ratio, ksize=1, padding=0, do_act=True,bias=bias)
        self.conv2 = ConvBnRelu3(channels//ratio, channels//ratio, ksize=3, padding=1, do_act=True,bias=bias)
        self.conv3 = ConvBnRelu3(channels//ratio, channels, ksize=1, padding=0, do_act=do_act,bias=bias)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out


class BottResidualBlock3(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ratio, num_convs,bias=True):
        super(BottResidualBlock3, self).__init__()
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvBnRelu3(channels, ratio, True,bias=bias))
            else:
                layers.append(BottConvBnRelu3(channels, ratio, False,bias=bias))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.ops(input)
        return self.act(input + output)


class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, padding, num_convs,bias=True):
        super(ResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=True,bias=bias))
            else:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=False,bias=bias))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        return self.act(input + output)        

class DownBlock(nn.Module):
    """ downsample block of v-net """

    def __init__(self, in_channels, num_convs, use_bottle_neck=False,bias = True):
        super(DownBlock, self).__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1, bias=bias)
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.down_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs, bias=bias)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs, bias=bias)

    def forward(self, input):
        out = self.down_act(self.down_bn(self.down_conv(input)))
        out = self.rblock(out)
        return out


class UpBlock(nn.Module):
    """ Upsample block of v-net """

    def __init__(self, in_channels, out_channels, num_convs, use_bottle_neck=False,bias=True):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1,bias=bias)
        self.up_bn = nn.BatchNorm3d(out_channels // 2)
        self.up_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels // 2, 4, num_convs,bias=bias)
        else:
            self.rblock = ResidualBlock3(out_channels // 2, 3, 1, num_convs,bias=bias)

    def forward(self, input, skip):
        out = self.up_act(self.up_bn(self.up_conv(input)))
#        out = torch.cat((out, skip), 1)
        out = torch.add(out, skip)
        out = self.rblock(out)
        return out


class OutputBlock(nn.Module):
    """ output block of v-net
    
        The output is a list of foreground-background probability vectors.
        The length of the list equals to the number of voxels in the volume
    """

    def __init__(self, in_channels, out_blocks):
        super(OutputBlock, self).__init__()
        self.out_blocks = out_blocks
        self.conv1 = nn.Conv3d(in_channels, out_blocks, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_blocks)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_blocks, out_blocks, kernel_size=1)
        tokens = torch.__version__.split('.')
        version = int(tokens[0] + tokens[1])
        if version < 3:
            self.softmax = nn.Softmax()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.act1(self.bn1(self.conv1(input)))
        out = self.conv2(out)
        out_size = out.size()
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(out.numel() // self.out_blocks, self.out_blocks)
        out = self.softmax(out)
        out = out.view(out_size[0], out_size[2], out_size[3], out_size[4], out_size[1])
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        return out


class Net(nn.Module):
    """ vnet for segmentation """
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1)
        self.down_64 = DownBlock(32, 2)
        self.down_128 = DownBlock(64, 3)
        self.down_256 = DownBlock(128, 3)
        self.up_128 = UpBlock(256, 256, 3)
        self.up_64 = UpBlock(256 // 2, 128, 3)
        self.up_32 = UpBlock(128 // 2, 64, 2)
        self.up_16 = UpBlock(64 // 2, 32, 1)
        self.out_block = OutputBlock(32 // 2, out_channels)
    @torch.cuda.amp.autocast() # 或 with 
    def forward(self, input):
        # 或 with torch.cuda.amp.autocast():
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        up128 = self.up_128(out256, out128)
        up64 = self.up_64(up128, out64)
        up32 = self.up_32(up64, out32)
        up16 = self.up_16(up32, out16)
        out = self.out_block(up16)
        return out

    def max_stride(self):
        return 16
