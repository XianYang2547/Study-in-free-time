# -*- coding: utf-8 -*-
# @Time    : 2023/8/30 15:49
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : calculate_mean_std.py
# ------❤️❤️❤️------ #


from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms


def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ImageFolder(root='train_val', transform=transforms.ToTensor())# 传参为父目录，下面必须得有子目录
    print(get_mean_and_std(train_dataset))
