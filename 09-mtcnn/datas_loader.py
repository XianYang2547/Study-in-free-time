# -*- coding: utf-8 -*-
# @Time    : 2023/7/4 13:15
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : datas_loader.py
# @SoftWare：PyCharm


from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np


# 12, 24 48
class Celeba_Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.dataset = []
        self.dataset.extend(open(f"{self.path}/positive.txt").readlines())
        self.dataset.extend(open(f"{self.path}/negative.txt").readlines())
        self.dataset.extend(open(f"{self.path}/part.txt").readlines())
        self.__len__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].split()
        # 置信度，偏移量
        # print(strs)
        conf = torch.Tensor([int(strs[1])])
        offset = torch.Tensor(
            [
                float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])
            ]
        )
        landmark = torch.Tensor([
            float(strs[i]) for i in range(6, 16)
        ])
        # print(landmark)
        # 返回：数据，置信度，建议框的偏移量
        # 图片路径
        img_path = f"{self.path}/{strs[0]}"
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255 - 0.5)
        # shape : HWC --> CHW
        img_data = img_data.permute(2, 0, 1)

        return img_data, conf, offset, landmark


if __name__ == '__main__':
    x = Celeba_Dataset(path=r'E:\xydataset\CelebA\sampledata\outimg\12')
    print(x)
