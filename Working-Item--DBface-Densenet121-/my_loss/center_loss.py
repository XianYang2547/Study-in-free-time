# -*- coding: utf-8 -*-
# @Time    : 2023/8/17 14:51
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : my_loss.py
# ------❤️❤️❤️------ #


import torch
import torch.nn as nn

data = torch.Tensor([
    [3, 1], [5, 9], [7, 2], [2, 4], [1, 5], [6, 8]
])
label = torch.Tensor([
    0, 0, 1, 0, 1, 0
])
center = torch.Tensor([
    [1, 2], [4, 4]
])
'''

# 根据label，将中心的扩为和数据一样形状，便于计算（x-y）    sqrt（sum（x-y）^2））
center_exp = torch.index_select(center, dim=0, index=label.long())
# exp = center.index_select(dim=0, index=label.long())

# 根据label统计各类标签的个数，便于后面用于除以平均距离【对标签分类】标签有几个类=bins，标签里面的最小值最大值
count = torch.histc(label, bins=2, min=0, max=1)

# 根据标签将标签类别数扩为跟数据一样的形状，便于 点减去中心的，再除以对于的个数，即平均距离
count_exp = torch.index_select(count, dim=0, index=label.long())

# 根据公式计算loss
my_loss = torch.sum(torch.div(torch.sqrt_(torch.sum(torch.pow(data - center_exp, 2), dim=1)), count_exp))
'''


# 封装为函数
def my_center_loss(data, label, center):
    # 根据label，将中心的扩为和数据一样形状，便于计算（x-y）    sqrt（sum（x-y）^2））
    center_exp = torch.index_select(center, dim=0, index=label.long())
    # 根据label统计各类标签的个数，便于后面用于除以平均距离【对标签分类】标签有几个类=bins，标签里面的最小值最大值
    count = torch.histc(label, bins=2, min=0, max=1)
    # 根据标签将标签类别数扩为跟数据一样的形状，便于 点减去中心的，再除以对于的个数，即平均距离
    count_exp = torch.index_select(count, dim=0, index=label.long())
    # 根据公式计算loss
    my_center_loss = torch.sum(torch.div(torch.sqrt_(torch.sum(torch.pow(data - center_exp, 2), dim=1)), count_exp))

    return my_center_loss


# loss = my_center_loss(data, label, center)

# 封装为类    因为中心的不易选择，定义为参数，让模型来学习得到中心点
class My_Center_Loss(nn.Module):
    def __init__(self, cls_num, feature_num):
        super().__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num, feature_num))

    def forward(self, data, label):
        # 根据label，将中心的扩为和数据一样形状，便于计算（x-y）    sqrt（sum（x-y）^2））
        center_exp = torch.index_select(self.center, dim=0, index=label.long())
        # 根据label统计各类标签的个数，便于后面用于除以平均距离【对标签分类】标签有几个类=bins，标签里面的最小值最大值
        count = torch.histc(label, bins=self.cls_num, min=0, max=self.cls_num - 1)
        # 根据标签将标签类别数扩为跟数据一样的形状，便于 点减去中心的，再除以对于的个数，即平均距离
        count_exp = torch.index_select(count, dim=0, index=label.long())
        # 根据公式计算loss
        my_center_loss = torch.sum(torch.div(torch.sqrt_(torch.sum(torch.pow(data - center_exp, 2), dim=1)), count_exp))
        return my_center_loss


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(784, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(2, 10)
        )

        self.center_loss_layer = My_Center_Loss(10, 2)  # 生成10个中心点
        self.crossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, xs):
        features = self.hidden_layer(xs)
        outputs = self.output_layer(features)
        return features, outputs

    def getloss(self, outputs, features, labels):
        loss_cls = self.crossEntropyLoss(outputs, labels)
        loss_center = self.center_loss_layer(features, labels)
        loss = loss_cls + loss_center
        return loss
