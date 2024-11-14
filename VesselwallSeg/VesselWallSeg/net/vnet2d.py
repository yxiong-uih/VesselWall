import torch.nn as nn
import torch
import numpy as np
from ...basic.kaiming_init import kaiming_weight_init
from ...basic.gaussian_init import gaussian_weight_init
from ...basic.xavier_init import xavier_weight_init
from torch.nn import functional as F


def vnet_kaiming_init(net):
    net.apply(kaiming_weight_init)

def vnet_xavier_init(net):
    net.apply(xavier_weight_init)

def vnet_focal_init(net, obj_p):
    net.apply(gaussian_weight_init)
    net.out_block.conv2.bias.data[1] = -np.log((1 - obj_p) / obj_p)


class Net1(nn.Module):
    """ v-net for segmentation """

    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        
        assert in_channels == 1,'vet2d only supports one input channel'
        
        self.conv_in128_chan16 = nn.Conv2d(in_channels, 16,kernel_size=5,stride=1,padding=2)
        #self.split1_1 = nn.ReplicationPad2d(2)
        self.outBlock1_1_RELU = nn.PReLU()
        self.down_pooling_128_to_64 = nn.Conv2d(16,32,2,2)
        self.relu_down_pooling_128_to_64 = nn.PReLU()
        self.conv_in64_chan32 = nn.Conv2d(32, 32,5,1,2)
        self.outBlock1_2_RELU = nn.PReLU()
        self.down_pooling_64_to_32 = nn.Conv2d(32, 64, 2,2)
        self.relu_down_pooling_64_to_32 = nn.PReLU()
        self.conv_in32_chan64 = nn.Conv2d(64, 64, 5,1,2)
        self.relu_conv_in32_chan64 = nn.PReLU()
        self.conv_in32_chan64_2 = nn.Conv2d(64,64,5,1,2)
        self.outBlock1_3_RELU = nn.PReLU()
        self.down_pooling_32_to_16= nn.Conv2d(64,128,2,2)
        self.relu_down_pooling_32_to_16 = nn.PReLU()
        self.conv_in16_chan128= nn.Conv2d(128,128,5,1,2)
        self.relu_conv_in16_chan128 = nn.PReLU()
        self.conv_in16_chan128_2= nn.Conv2d(128,128,5,1,2)
        self.relu_conv_in16_chan128_2 = nn.PReLU()
        self.conv_in16_chan128_3= nn.Conv2d(128,128,5,1,2)
        self.outBlock1_4_RELU = nn.PReLU()
        self.down_pooling_16_to_8= nn.Conv2d(128,256,2,2)
        self.relu_down_pooling_16_to_8 = nn.PReLU()
        self.conv_in8_chan256= nn.Conv2d(256,256,5,1,2)
        self.relu_conv_in8_chan256 = nn.PReLU()

        self.conv_in8_chan256_2= nn.Conv2d(256,256,5,1,2)
        self.relu_conv_in8_chan256_2 = nn.PReLU()
        self.conv_in8_chan256_3= nn.Conv2d(256,256,5,1,2)
        self.outBlock1_5_RELU = nn.PReLU()
        self.deconv_in8_chan128 = nn.ConvTranspose2d(256,128,2,2)
        self.relu_deconv_in8_chan128 = nn.PReLU()
        #self.concat_in16_concat = torch.cat(2)
        self.conv_in16_chan128_right= nn.Conv2d(256,256,5,1,2)
        self.relu_conv_in16_chan128_right = nn.PReLU()
        self.conv_in16_chan128_right_2= nn.Conv2d(256,256,5,1,2)
        self.relu_conv_in16_chan128_right_2 = nn.PReLU()
        self.conv_in16_chan128_right_3= nn.Conv2d(256,256,5,1,2)
        self.outBlock2_4_RELU = nn.PReLU()
        self.deconv_in16_chan64 = nn.ConvTranspose2d(256,64,2,2)
        self.relu_deconv_in16_chan64 = nn.PReLU()
        #self.concat_in32_concat = torch.cat(2)
        self.conv_in32_chan64_right = nn.Conv2d(128,128,5,1,2)
        self.relu_conv_in32_chan64_right = nn.PReLU()
        self.conv_in32_chan64_right_2 = nn.Conv2d(128,128,5,1,2)
        self.outBlock2_3_RELU = nn.PReLU()
        self.deconv_in32_chan32 = nn.ConvTranspose2d(128,32,2,2)
        self.relu_deconv_in32_chan32 = nn.PReLU()
        #self.concat_in64_concat = torch.cat(2)
        self.conv_in64_chan32_right= nn.Conv2d(64,64,5,1,2)
        self.outBlock2_2_RELU = nn.PReLU()
        self.deconv_in64_chan16 = nn.ConvTranspose2d(64,16,2,2)
        self.relu_deconv_in64_chan16 = nn.PReLU()
        #self.concat_in128_concat = torch.cat(2)
        self.conv_in128_chan16_right= nn.Conv2d(32,32,5,1,2)
        self.outBlock2_1_RELU = nn.PReLU()
        self.conv_in128_chan2_right= nn.Conv2d(32,16,5,1,2)
        self.relu_conv_in128_chan2_right = nn.PReLU()
        self.conv_in128_chan2_2_right= nn.Conv2d(16,out_channels,1,1)
        self.softmax = nn.Softmax(1)

    def forward(self, input):


        m_conv_in128_chan16_out = self.conv_in128_chan16(input)
        m_split1_1_out = input.repeat(1, 16, 1, 1)
        m_outBlock1_1_RELU_out = self.outBlock1_1_RELU(m_conv_in128_chan16_out + m_split1_1_out)
        m_down_pooling_128_to_64_out = self.down_pooling_128_to_64(m_outBlock1_1_RELU_out)
        m_relu_down_pooling_128_to_64_out = self.relu_down_pooling_128_to_64(m_down_pooling_128_to_64_out)
        m_conv_in64_chan32_out = self.conv_in64_chan32(m_relu_down_pooling_128_to_64_out)
        m_outBlock1_2_RELU_out = self.outBlock1_2_RELU(m_conv_in64_chan32_out + m_relu_down_pooling_128_to_64_out)
        m_down_pooling_64_to_32_out = self.down_pooling_64_to_32(m_outBlock1_2_RELU_out)
        m_relu_down_pooling_64_to_32_out = self.relu_down_pooling_64_to_32(m_down_pooling_64_to_32_out)

        m_conv_in32_chan64_out = self.conv_in32_chan64(m_relu_down_pooling_64_to_32_out)
        m_relu_conv_in32_chan64_out = self.relu_conv_in32_chan64(m_conv_in32_chan64_out)
        m_conv_in32_chan64_2_out = self.conv_in32_chan64_2(m_relu_conv_in32_chan64_out)
        m_outBlock1_3_RELU_out = self.outBlock1_3_RELU(m_conv_in32_chan64_2_out + m_relu_down_pooling_64_to_32_out)
        m_down_pooling_32_to_16_out = self.down_pooling_32_to_16(m_outBlock1_3_RELU_out)
        m_relu_down_pooling_32_to_16_out = self.relu_down_pooling_32_to_16(m_down_pooling_32_to_16_out)

        m_conv_in16_chan128_out = self.conv_in16_chan128(m_relu_down_pooling_32_to_16_out)
        m_relu_conv_in16_chan128_out = self.relu_conv_in16_chan128(m_conv_in16_chan128_out)
        m_conv_in16_chan128_2_out = self.conv_in16_chan128_2(m_relu_conv_in16_chan128_out)
        m_relu_conv_in16_chan128_2_out = self.relu_conv_in16_chan128_2(m_conv_in16_chan128_2_out)
        m_conv_in16_chan128_3_out = self.conv_in16_chan128_3(m_relu_conv_in16_chan128_2_out)
        m_outBlock1_4_RELU_out = self.outBlock1_4_RELU(m_conv_in16_chan128_3_out + m_relu_down_pooling_32_to_16_out)
        m_down_pooling_16_to_8_out = self.down_pooling_16_to_8(m_outBlock1_4_RELU_out)
        m_relu_down_pooling_16_to_8_out = self.relu_down_pooling_16_to_8(m_down_pooling_16_to_8_out)
        m_conv_in8_chan256_out = self.conv_in8_chan256(m_relu_down_pooling_16_to_8_out)
        m_relu_conv_in8_chan256_out = self.relu_conv_in8_chan256(m_conv_in8_chan256_out)
        m_conv_in8_chan256_2_out = self.conv_in8_chan256_2(m_relu_conv_in8_chan256_out)
        m_relu_conv_in8_chan256_2_out = self.relu_conv_in8_chan256_2(m_conv_in8_chan256_2_out)
        m_conv_in8_chan256_3_out = self.conv_in8_chan256_3(m_relu_conv_in8_chan256_2_out)
        m_outBlock1_5_RELU_out = self.outBlock1_5_RELU(m_conv_in8_chan256_3_out + m_relu_down_pooling_16_to_8_out)
        m_deconv_in8_chan128_out = self.deconv_in8_chan128(m_outBlock1_5_RELU_out)
        m_relu_deconv_in8_chan128_out = self.relu_deconv_in8_chan128(m_deconv_in8_chan128_out)
        m_concat_in16_concat_out = torch.cat((m_relu_deconv_in8_chan128_out,m_outBlock1_4_RELU_out),1)
        m_conv_in16_chan128_right_out = self.conv_in16_chan128_right(m_concat_in16_concat_out)
        m_relu_conv_in16_chan128_right_out = self.relu_conv_in16_chan128_right(m_conv_in16_chan128_right_out)
        m_conv_in16_chan128_right_2_out = self.conv_in16_chan128_right_2(m_relu_conv_in16_chan128_right_out)
        m_relu_conv_in16_chan128_right_2_out = self.relu_conv_in16_chan128_right_2(m_conv_in16_chan128_right_2_out)
        m_conv_in16_chan128_right_3_out = self.conv_in16_chan128_right_3(m_relu_conv_in16_chan128_right_2_out)
        m_outBlock2_4_RELU_out = self.outBlock2_4_RELU(m_conv_in16_chan128_right_3_out + m_concat_in16_concat_out)
        m_deconv_in16_chan64_out = self.deconv_in16_chan64(m_outBlock2_4_RELU_out)
        m_relu_deconv_in16_chan64_out = self.relu_deconv_in16_chan64(m_deconv_in16_chan64_out)
        m_concat_in32_concat_out = torch.cat((m_relu_deconv_in16_chan64_out,m_outBlock1_3_RELU_out),1)
        m_conv_in32_chan64_right_out = self.conv_in32_chan64_right(m_concat_in32_concat_out)
        m_relu_conv_in32_chan64_right_out = self.relu_conv_in32_chan64_right(m_conv_in32_chan64_right_out)
        m_conv_in32_chan64_right_2_out = self.conv_in32_chan64_right_2(m_relu_conv_in32_chan64_right_out)
        m_outBlock2_3_RELU_out = self.outBlock2_3_RELU(m_conv_in32_chan64_right_2_out + m_concat_in32_concat_out)
        m_deconv_in32_chan32_out = self.deconv_in32_chan32(m_outBlock2_3_RELU_out)
        m_relu_deconv_in32_chan32_out = self.relu_deconv_in32_chan32(m_deconv_in32_chan32_out)
        m_concat_in64_concat_out = torch.cat((m_relu_deconv_in32_chan32_out,m_outBlock1_2_RELU_out),1)
        m_conv_in64_chan32_right_out = self.conv_in64_chan32_right(m_concat_in64_concat_out)
        m_outBlock2_2_RELU_out = self.outBlock2_2_RELU(m_conv_in64_chan32_right_out + m_concat_in64_concat_out)
        m_deconv_in64_chan16_out = self.deconv_in64_chan16(m_outBlock2_2_RELU_out)
        m_relu_deconv_in64_chan16_out = self.relu_deconv_in64_chan16(m_deconv_in64_chan16_out)
        m_concat_in128_concat_out = torch.cat((m_relu_deconv_in64_chan16_out,m_outBlock1_1_RELU_out),1)
        m_conv_in128_chan16_right_out = self.conv_in128_chan16_right(m_concat_in128_concat_out)
        m_outBlock2_1_RELU_out = self.outBlock2_1_RELU(m_conv_in128_chan16_right_out + m_concat_in128_concat_out)
        m_conv_in128_chan2_right_out = self.conv_in128_chan2_right(m_outBlock2_1_RELU_out)
        m_relu_conv_in128_chan2_right_out = self.relu_conv_in128_chan2_right(m_conv_in128_chan2_right_out)
        m_conv_in128_chan2_2_right_out = self.conv_in128_chan2_2_right(m_relu_conv_in128_chan2_right_out)

        return self.softmax(m_conv_in128_chan2_2_right_out)

    def max_stride(self):
        return 16

class Attention_block(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super(Attention_block, self).__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)

            return x * psi

def contracting_block(in_channels, out_channels):
        block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

double_conv = contracting_block  # 上采样过程中也反复使用了两层卷积的结构

class expansive_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(expansive_block, self).__init__()

            self.block = nn.Sequential(
                nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )

        def forward(self, x):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            out = self.block(x)
            return out

def final_block(in_channels, out_channels):
        return nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0)

class Net(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Net, self).__init__()
        # Encode
        self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=16)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = contracting_block(in_channels=16, out_channels=32)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = contracting_block(in_channels=32, out_channels=64)
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = contracting_block(in_channels=64, out_channels=128)
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode5 = contracting_block(in_channels=128, out_channels=256)

        # Decode
        self.conv_decode4 = expansive_block(256, 128)
        self.att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.double_conv4 = double_conv(256, 128)

        self.conv_decode3 = expansive_block(128, 64)
        self.att3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.double_conv3 = double_conv(128, 64)

        self.conv_decode2 = expansive_block(64, 32)
        self.att2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.double_conv2 = double_conv(64, 32)

        self.conv_decode1 = expansive_block(32, 16)
        self.att1 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.double_conv1 = double_conv(32, 16)

        self.final_layer = final_block(16, out_channel)
        self.softmax = nn.Softmax(1)

    def forward(self, input):
        # Encode
        encode_block1 = self.conv_encode1(input);
        encode_pool1 = self.conv_pool1(encode_block1);
        encode_block2 = self.conv_encode2(encode_pool1);
        encode_pool2 = self.conv_pool2(encode_block2);
        encode_block3 = self.conv_encode3(encode_pool2);
        encode_pool3 = self.conv_pool3(encode_block3);
        encode_block4 = self.conv_encode4(encode_pool3);
        encode_pool4 = self.conv_pool4(encode_block4);
        encode_block5 = self.conv_encode5(encode_pool4);

        # Decode
        decode_block4 = self.conv_decode4(encode_block5)
        encode_block4 = self.att4(g=decode_block4, x=encode_block4)
        decode_block4 = torch.cat((encode_block4, decode_block4), dim=1)
        decode_block4 = self.double_conv4(decode_block4);

        decode_block3 = self.conv_decode3(encode_block4)
        encode_block3 = self.att3(g=decode_block3, x=encode_block3)
        decode_block3 = torch.cat((encode_block3, decode_block3), dim=1)
        decode_block3 = self.double_conv3(decode_block3);

        decode_block2 = self.conv_decode2(encode_block3)
        encode_block2 = self.att2(g=decode_block2, x=encode_block2)
        decode_block2 = torch.cat((encode_block2, decode_block2), dim=1)
        decode_block2 = self.double_conv2(decode_block2);

        decode_block1 = self.conv_decode1(encode_block2)
        encode_block1 = self.att1(g=decode_block1, x=encode_block1)
        decode_block1 = torch.cat((encode_block1, decode_block1), dim=1)
        decode_block1 = self.double_conv1(decode_block1);


        final_layer = self.final_layer(decode_block1)
        return self.softmax(final_layer)

    def max_stride(self):
        return 16