#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 15:11
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : my_dataset.py


from my_utils import img_pro, one_hot
from torch.utils.data import Dataset

"先对原始数据进行自适应直方图均衡化，然后旋转增样"
"调用旋转后，进行数据集划分"


class My_data(Dataset):
    def __init__(self, arthrosis, fc):
        "arthrosis,fc作为变量方便加载不同目录下的不同文件"
        super().__init__()
        self.arthrosis = arthrosis
        self.path = r"E:\xydataset\bone_age\arthrosis"
        self.data = open(f"{self.path}\\{arthrosis[0]}\\{fc}.txt").readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        imagepath, label = data.split(' ')
        image = img_pro(imagepath)
        label = one_hot(self.arthrosis[1], label)
        return image, label


if __name__ == '__main__':
    arthrosises = {'MCPFirst': ['MCPFirst', 11],  # 第一手指掌骨
                   'DIPFirst': ['DIPFirst', 11],  # 第一手指远节指骨
                   'PIPFirst': ['PIPFirst', 12],  # 第一手指近节指骨
                   'MIP': ['MIP', 12],  # 中节指骨（除了拇指剩下四只手指）（第一手指【拇指】是没有中节指骨的））
                   'Radius': ['Radius', 14],  # 桡骨
                   'Ulna': ['Ulna', 12],  # 尺骨
                   'PIP': ['PIP', 12],  # 近节指骨（除了拇指剩下四只手指）
                   'DIP': ['DIP', 11],  # 远节指骨（除了拇指剩下四只手指）
                   'MCP': ['MCP', 10]}  # 掌骨（除了拇指剩下四只手指）
    from torchvision import models

    mo = models.resnet18(num_classes=10)
    print(mo)
    # data = DataLoader(My_data(arthrosises['MIP'], 'train'), 1, shuffle=True)
    # for i, (img, label) in enumerate(data):
    #     print(label)
    #     exit()
