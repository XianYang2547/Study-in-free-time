# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 16:36
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : VOC_Dataset.py
# @SoftWare：PyCharm


from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

transform = transforms.Compose([
    transforms.ToTensor()
])


class VOC_Dataset(Dataset):
    """取出原数据集里面的2913个分割标签和对应的图像文件用来测试"""

    def __init__(self, path=r"E:\xydataset\VOC2012"):
        super().__init__()
        self.path = path
        # 标签 SegmentationClass
        self.label_names = os.listdir(f"{self.path}/SegmentationClass")

    def __len__(self):
        return len(self.label_names)

    def __getitem__(self, index):
        # 从标签文件构建数据文件
        # 等比例压缩，粘贴
        bg1 = transforms.ToPILImage()(torch.zeros(3, 256, 256))
        bg2 = transforms.ToPILImage()(torch.zeros(3, 256, 256))
        # 获取标签图像名称
        name = self.label_names[index]
        # 拿出标签文件名（不要后缀），添加新的后缀构成新的名字来和训练图像对应
        name_jpg = name[:-3] + "jpg"
        # JPEGImages
        img1_path = f"{self.path}/JPEGImages"
        img2_path = f"{self.path}/SegmentationClass"
        # 构建完整数据路径和名字
        img1 = Image.open(f"{img1_path}/{name_jpg}")
        img2 = Image.open(f"{img2_path}/{name}")
        # 等比例压缩
        img1_size = torch.Tensor(img1.size)
        # 压缩率
        scale = 256 / img1.size[img1_size.argmax()]
        img_scale = (img1_size * scale).long()
        img1 = img1.resize(img_scale)
        img2 = img2.resize(img_scale)
        # 背景粘贴
        bg1.paste(img1, (0, 0))
        bg2.paste(img2, (0, 0))
        return transform(bg1), transform(bg2)


if __name__ == '__main__':
    # 对处理的数据取一些 出来可视化
    voc_ds = VOC_Dataset()
    i = 1
    for data, label in voc_ds:
        save_image(data, f"E:\\xydataset\\VOC2012\\Unet_train_voc\train\\{i}.jpg", nrow=1)
        save_image(label, f"E:\\xydataset\\VOC2012\\Unet_train_voc\\label\\{i}.png", nrow=1)
        i += 1
