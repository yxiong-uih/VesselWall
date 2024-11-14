# for rknn conv

"""Fast Segmentation Convolutional Neural Network"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNN']

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out
    

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        # self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)
        #self.out = _ConvBNReLU(224, out_channels, 1)
        self.out = _ConvBNReLU(inter_channels*7, out_channels, 1)
        
        self.avgpool1 = nn.AvgPool2d(1)
        self.avgpool2 = nn.AvgPool2d(2)
        self.avgpool3 = nn.AvgPool2d(3)
        # self.avgpool6 = nn.AvgPool2d(6)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.avgpool1(x)), size)
        feat2 = self.upsample(self.conv2(self.avgpool2(x)), size)
        feat3 = self.upsample(self.conv3(self.avgpool3(x)), size)
        # feat4 = self.upsample(self.conv4(self.avgpool6(x)), size)
        # x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = torch.cat([x, feat1, feat2, feat3], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, inputs_channel, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(inputs_channel, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class GlobalFeatureExtractor3(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 128),
                 out_channels=128, t=6, num_blocks=(3, 3), **kwargs):
        super(GlobalFeatureExtractor3, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.ppm = PyramidPooling(block_channels[1], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x



class FastSCNN(nn.Module):
    def __init__(self, inputs_channel, outputs_channel, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(inputs_channel, 32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, outputs_channel)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, outputs_channel, 1)
            )
    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        # for training
        x = F.softmax(x, dim=1)
        # for output
        # del x = F.softmax(x, dim=1)
        # x = x.argmax(1)

        # if self.aux:
        #     auxout = self.auxlayer(higher_res_features)
        #     auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
        #     x      = auxout
        return x
    

class FastSCNN_UT(nn.Module):
    def __init__(self, inputs_channel, outputs_channel, **kwargs):
        super(FastSCNN_UT, self).__init__()
        self.learning_to_downsample     = LearningToDownsample(inputs_channel, 16, 24, 32)
        self.global_feature_extractor   = GlobalFeatureExtractor3(32, [32, 64], 64, 2, [3, 3])
        self.feature_fusion             = FeatureFusionModule(32, 64, 64)
        self.classifier                 = Classifer(64, outputs_channel)
        
        self.upsampler                  = torch.nn.Upsample(scale_factor=8, mode='nearest') # https://zhuanlan.zhihu.com/p/87572724
        # self.upsampler                  = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # https://zhuanlan.zhihu.com/p/87572724
        self.kwargs                     = kwargs

    def forward(self, x):

        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
                                        
        if self.kwargs.get("UT")==1:    # UT_1 : upsampler + softmax    输出为 NCHW_1x3x128x256     -   用于训练
            x = self.upsampler(x)
            x = F.softmax(x, dim=1)
            return x
                                        
        elif self.kwargs.get("UT")==2:  # UT_2 : upsampler              输出为 NCHW_1x3x128x256     -   用于导出
            x = self.upsampler(x)
            return x
                                        
        elif self.kwargs.get("UT")==3:  # UT_3 : upsampler + argmax     输出为 NCHW_1x128x256       -   用于导出
            x = self.upsampler(x)
            x = x.argmax(1)
            return x
        
        elif self.kwargs.get("UT")==4:  # UT_4 : argmax                 输出为 NCHW_1x16x32         -   用于导出
            x = x.argmax(1)
            return x


def FastSCNN_UT_UT():
    img = torch.randn(1, 3, 128, 256)
    model = FastSCNN_UT(inputs_channel=3, outputs_channel=3, UT=2)
    outputs = model(img)
    print(outputs.shape)




''''''''''''''''' 专门为 rknn 做的卷积 '''''''''''''''''
class _ConvBNReLU_RKNN(nn.Module):
    """Conv-BN-ReLU RKNN"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class LearningToDownsample_RKNN(nn.Module):
    """Learning to downsample module"""

    def __init__(self, in_channels, mid_channels_1=32, mid_channels_2=48, out_channels=64, **kwargs):
        super().__init__()
        self.conv1 = _ConvBNReLU_RKNN(in_channels,    mid_channels_1, 3, 2)
        self.conv2 = _ConvBNReLU_RKNN(mid_channels_1, mid_channels_2, 3, 2)
        self.conv3 = _ConvBNReLU_RKNN(mid_channels_2,   out_channels, 3, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print("LearningToDownsample_RKNN")
        # print(x.shape)
        return x

class LinearBottleneck_RKNN(nn.Module):
    """LinearBottleneck used in MobileNetV2"""
    def __init__(self, in_channels, out_channels, t=3, stride=2, **kwargs):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        # CBA+CB
        self.block = nn.Sequential(
            _ConvBNReLU_RKNN(in_channels, in_channels * t, 3, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, 1, bias=False), # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class GlobalFeatureExtractor_RKNN(nn.Module):
    """Global feature extractor module"""
    def __init__(self, in_channels=32, block_channels=(32, 48, 64), out_channels=64, t=2, num_blocks=(3, 3, 3), **kwargs):
        super().__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck_RKNN, in_channels,         block_channels[0], t, 2, num_blocks[0])
        self.bottleneck2 = self._make_layer(LinearBottleneck_RKNN, block_channels[0],   block_channels[1], t, 2, num_blocks[1])
        self.bottleneck3 = self._make_layer(LinearBottleneck_RKNN, block_channels[1],   out_channels,      t, 1, num_blocks[2])

    def _make_layer(self, block, inplanes, planes, t, stride, blocks):
        layers = [block(inplanes, planes, t, stride)]
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        # print("GlobalFeatureExtractor_RKNN")
        # print(x.shape)
        return x



class FeatureFusionModule_RKNN(nn.Module):
    """ Feature fusion module - 用于收集信息 """
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super().__init__()

        # x8
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # x16
        self.upsampler      = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv_lower_res = nn.Sequential(
            # _ConvBNReLU_RKNN(lower_in_channels, out_channels, 3, 1),
            nn.Conv2d(lower_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu           = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        
        # x16 -> x8 -> CB
        lower_res_feature = self.upsampler(lower_res_feature)           # x16 -> x8size
        lower_res_feature = self.conv_lower_res(lower_res_feature)      # ******* 新加CB *******
        
        # x8 -> CB
        higher_res_feature = self.conv_higher_res(higher_res_feature)   # x8size

        # add
        out = lower_res_feature + higher_res_feature
        # print("FeatureFusionModule_RKNN")
        # print(lower_res_feature.shape)
        # print(higher_res_feature.shape)
        # print(out.shape)
        return self.relu(out)
    

class Classifer_RKNN(nn.Module):
    """Classifer RKNN - 用于优化边缘 """

    def __init__(self, in_channels, out_classes, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU_RKNN(in_channels, in_channels, 3, 1),
            # _ConvBNReLU_RKNN(in_channels, in_channels, 3, 1),               # ******* 新加CBA - loss降得快，精度提得慢？*******
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, out_classes, 1),
            # nn.BatchNorm2d(out_classes)                                   # ******* 新加B - 貌似没啥用 *******
        )

    def forward(self, x):
        x = self.conv(x)
        return x    


class FastSCNN_RKNN(nn.Module):
    def __init__(self, inputs_channel, outputs_channel, **kwargs):
        super().__init__()
        self.learning_to_downsample     = LearningToDownsample_RKNN(inputs_channel, 16, 24, 32)
        self.global_feature_extractor   = GlobalFeatureExtractor_RKNN(32, [32, 48, 64], 64, 2, [3, 3, 3]) # t=4->2 
        self.feature_fusion_module      = FeatureFusionModule_RKNN(32, 64, 64)
        self.classifer                  = Classifer_RKNN(64, outputs_channel)
        self.upsampler                  = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # https://zhuanlan.zhihu.com/p/87572724
        self.kwargs                     = kwargs

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features  = self.global_feature_extractor(higher_res_features)
        x                   = self.feature_fusion_module(higher_res_features, lower_res_features)
        x                   = self.classifer(x)

        if self.kwargs.get("UT")==5:        # UT_5 : upsampler + softmax       输出为 NCHW_1x3x128x256  -   用于训练
            x = self.upsampler(x)
            x = F.softmax(x, dim=1)
            return x
        # elif self.kwargs.get("UT")==6:      # UT_6 : argmax                    输出为 NCHW_1x16x32      -   用于导出
        #     x = x.argmax(1)
        #     return x

        elif self.kwargs.get("UT")==7:      # UT_7 : upsampler + argmax        输出为 NCHW_1x128x256    -   用于导出
            x = self.upsampler(x)
            x = x.argmax(1)
            return x



class Net(nn.Module):
    def __init__(self, inputs_channel, outputs_channel):
        super().__init__()
        self.learning_to_downsample     = LearningToDownsample_RKNN(inputs_channel, 16, 24, 32)
        self.global_feature_extractor   = GlobalFeatureExtractor_RKNN(32, [32, 48, 64], 64, 2, [3, 3, 3]) # t=4->2 
        self.feature_fusion_module      = FeatureFusionModule_RKNN(32, 64, 64)
        self.classifer                  = Classifer_RKNN(64, outputs_channel)
        self.upsampler                  = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # https://zhuanlan.zhihu.com/p/87572724

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features  = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion_module(higher_res_features, lower_res_features)
        x = self.classifer(x)
        x = self.upsampler(x)
        x = F.softmax(x, dim=1)
        return x


def FastSCNN_RKNN_UT():
    inputs  = torch.randn(1, 3, 128, 256)
    model   = FastSCNN_RKNN(inputs_channel=3, outputs_channel=3, UT=7)
    outputs = model(inputs)
    
    print("FastSCNN_RKNN_UT")
    print(outputs.shape)



if __name__ == '__main__':

    FastSCNN_RKNN_UT()

    # FastSCNN_RKNN_EXP_UT()

    # FastSCNN_UT_UT()
