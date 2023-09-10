# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 21:00
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : MyUnet.py
# @SoftWare：PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    # unet中2个卷积层，挨在一起的
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(  # 使用反射填充
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # 使用3*3卷积步长为2来代替下采样的最大池化操作，避免丢信息
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(channel),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSampling(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Sequential(
            # 使用1*1卷积改变通道
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1)
        )

    def forward(self, x, featuremap):
        # 上采样：插值法
        out = F.interpolate(x, scale_factor=2, mode='nearest')
        # 上采样后经过一个卷积,通道减半
        out = self.layer(out)
        # 拼接
        return torch.cat((out, featuremap), dim=1)


class MyUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 下采样
        self.conv1 = Conv_Block(3, 64)
        self.down1 = DownSampling(64)
        self.conv2 = Conv_Block(64, 128)
        self.down2 = DownSampling(128)
        self.conv3 = Conv_Block(128, 256)
        self.down3 = DownSampling(256)
        self.conv4 = Conv_Block(256, 512)
        self.down4 = DownSampling(512)
        self.conv5 = Conv_Block(512, 1024)
        # 上采用
        self.up1 = UpSampling(1024)
        self.conv6 = Conv_Block(1024, 512)
        self.up2 = UpSampling(512)
        self.conv7 = Conv_Block(512, 256)
        self.up3 = UpSampling(256)
        self.conv8 = Conv_Block(256, 128)
        self.up4 = UpSampling(128)
        self.conv9 = Conv_Block(128, 64)
        # 1*1卷积输出
        self.out = nn.Conv2d(64, 3, kernel_size=1, stride=1)

    def forward(self, x):
        # 左边的前向计算
        out1 = self.conv1(x)
        out2 = self.conv2(self.down1(out1))
        out3 = self.conv3(self.down2(out2))
        out4 = self.conv4(self.down3(out3))
        out5 = self.conv5(self.down4(out4))
        # 右边的前向计算 上采样包含卷积和拼接，然后经过卷积块
        out6 = self.conv6(self.up1(out5, out4))
        out7 = self.conv7(self.up2(out6, out3))
        out8 = self.conv8(self.up3(out7, out2))
        out9 = self.conv9(self.up4(out8, out1))
        return self.out(out9)


if __name__ == '__main__':
    # HW
    data = torch.randn(1, 3, 256, 256)
    net = MyUnet()
    data = net(data)
    print(data.shape)
    model = MyUnet()
    params = model.state_dict()
    total_params = sum(p.numel() for p in params.values())
    print(f'The total number of parameters is {total_params}')
