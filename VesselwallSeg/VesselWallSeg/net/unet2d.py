import torch.nn as nn
import torch
import numpy as np
from ...basic.kaiming_init import kaiming_weight_init
from ...basic.gaussian_init import gaussian_weight_init
from ...basic.xavier_init import xavier_weight_init
import torch.nn.functional as F

def vnet_kaiming_init(net):
    net.apply(kaiming_weight_init)

def vnet_focal_init(net, obj_p):
    net.apply(gaussian_weight_init)
    net.out_block.conv2.bias.data[1] = -np.log((1 - obj_p) / obj_p)

def vnet_xavier_init(net):
    net.apply(xavier_weight_init)


class Net(nn.Module):
    """
    2d U-Net
    references：
    [1] 2d U-Net Learning Dense Volumetric Segmentation from Sparse Annotation MICCAI 2016
    [2] U-Net Convolutional Networks for Biomedical Image Segmentation MICCAI 2015
    """

    def __init__(self, in_channels=1, out_channels=3):
        """
        """
        super(Net, self).__init__()
        self.inconv = inconv(in_channels, 8)
        self.downsample1 = downsample(16, 16)
        self.downsample2 = downsample(32, 32)
        self.downsample3 = downsample(64, 64)
        self.upsample1 = upsample(128, 64)
        self.upsample2 = upsample(64, 32)
        self.upsample3 = upsample(32, 16)
        self.outconv = outconv(16, out_channels)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x = self.upsample1(x4, x3)
        x = self.upsample2(x, x2)
        x = self.upsample3(x, x1)
        x = self.outconv(x)
        return self.softmax(x)
        #return x

    def max_stride(self):
        return 8


class inconv(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        """
        """
        super(inconv, self).__init__()
        self.conv = VGGConvEncoder(inChannels, outChannels)

    def forward(self, x):
        x = self.conv(x)
        return x


class VGGConvEncoder1(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        """
        """
        super(VGGConvEncoder, self).__init__()
        self.VGGConv = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outChannels, out_channels=outChannels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.VGGConv(x)
        return x

class VGGConvEncoder(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        """
        """
        super(VGGConvEncoder, self).__init__()
        self.VGGConv1= nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels * 2, kernel_size=1),
            nn.BatchNorm2d(outChannels * 2),
            nn.ReLU(inplace=True),
        )
        self.VGGConv2 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outChannels, out_channels=outChannels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels * 2),
            nn.ReLU(inplace=True)
        )
        self.VGGConv3 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outChannels, out_channels=outChannels * 2, kernel_size=5, padding=1),
            nn.BatchNorm2d(outChannels * 2),
            nn.ReLU(inplace=True)
        )
        self.VGGConv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels * 2, kernel_size=1),
            nn.BatchNorm2d(outChannels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.VGGConv1(x)
        x2 = self.VGGConv2(x)
        x3 = self.VGGConv3(x)
        x4 = self.VGGConv4(x)
        x = x1 + x2 + x3 + x4;
        return x


class VGGConvDecoder1(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        """
        """
        super(VGGConvDecoder, self).__init__()
        self.VGGConv = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outChannels, out_channels=outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.VGGConv(x)
        return x

class VGGConvDecoder(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        """
        """
        super(VGGConvDecoder, self).__init__()
        self.VGGConv1 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )
        self.VGGConv2 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outChannels, out_channels=outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )
        self.VGGConv3 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outChannels, out_channels=outChannels, kernel_size=5, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )
        self.VGGConv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.VGGConv1(x)
        x2 = self.VGGConv2(x)
        x3 = self.VGGConv3(x)
        x4 = self.VGGConv4(x)
        x = x1 + x2 + x3 + x4;
        return x

class downsample(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        """
        """
        super(downsample, self).__init__()
        self.MPConv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            VGGConvEncoder(inChannels, outChannels)
        )

    def forward(self, x):
        x = self.MPConv(x)
        return x


class upsample(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        """
        """
        super(upsample, self).__init__()
        self.TRConv = nn.ConvTranspose2d(inChannels, outChannels, kernel_size=2, stride=2)

        self.conv = VGGConvDecoder(inChannels, outChannels)

    def forward(self, x1, x2):
        x1 = self.TRConv(x1)
        #diffZ = x1.size()[2] - x2.size()[2]  # depth, z
        #diffY = x1.size()[3] - x2.size()[3]  # height, y
        #diffX = x1.size()[4] - x2.size()[4]  # width, x
        diffY = x1.size()[2] - x2.size()[2]  # height, y
        diffX = x1.size()[3] - x2.size()[3]  # width, x
        # refer to https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        #x2 = F.pad(x2, (diffZ // 2, int(diffZ / 2), diffY // 2, int(diffY / 2), diffX // 2, int(diffX / 2)))
        x2 = F.pad(x2,  (diffY // 2, int(diffY / 2), diffX // 2, int(diffX / 2)))
        # refer to https://pytorch.org/docs/stable/torch.html#torch.cat
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    """
    """

    def __init__(self, inChannels, outChannels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x