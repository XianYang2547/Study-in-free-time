# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 16:36
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : VOC_Dataset.py
# @SoftWare：PyCharm
import torchvision
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
        bg1 = transforms.ToPILImage()(torch.zeros(3, 128, 128))
        bg2 = transforms.ToPILImage()(torch.zeros(1, 128, 128))
        # 获取标签名
        # 2007_000032.png --> 2007_000032.jpg
        name = self.label_names[index]
        # 2007_000032 --> 2007_000032.jpg
        name_jpg = name[:-3] + "jpg"
        # JPEGImages
        img1_path = f"{self.path}/JPEGImages"
        img2_path = f"{self.path}/SegmentationClass"
        # 数据
        img1 = Image.open(f"{img1_path}/{name_jpg}")
        # 标签 处理成单通道的，适应u2net的输出
        img2 = Image.open(f"{img2_path}/{name}")
        img2 = img2.convert("L")  # 使用unet时请注释它
        # 等比例压缩
        img1_size = torch.Tensor(img1.size)
        # 压缩率
        scale = 128 / img1.size[img1_size.argmax()]
        img_scale = (img1_size * scale).long()
        img1 = img1.resize(img_scale)
        img2 = img2.resize(img_scale)
        # 背景粘贴
        bg1.paste(img1, (0, 0))
        bg2.paste(img2, (0, 0))
        return transform(bg1), transform(bg2)


class my_vocdata(Dataset):
    """从上面的类生成的图像，自己进行划分
    并加载测试集
    """

    def __init__(self, istest=False):
        super().__init__()
        self.istest = istest
        self.train_root = r"E:\xydataset\VOC2012\Unet_train_voc\train"
        self.train_label = r"E:\xydataset\VOC2012\Unet_train_voc\label"
        self.test_root = r"E:\xydataset\VOC2012\Unet_train_voc\test"
        # self.__getitem__(1)

    def __getitem__(self, item):
        if not self.istest:
            train_img = f"{self.train_root}\\{os.listdir(self.train_root)[item]}"
            train_lab = f"{self.train_label}\\{os.listdir(self.train_label)[item]}"
            img = Image.open(train_img)
            lab = Image.open(train_lab)
            lab = lab.convert("L")
            return transform(img), transform(lab),
        else:
            test_image = f"{self.test_root}\\{os.listdir(self.test_root)[item]}"
            test_img = Image.open(test_image)
            test_img = test_img.resize((256, 256))
            return transform(test_img)

    def __len__(self):
        return len(os.listdir(self.train_root)) if not self.istest else len(os.listdir(self.test_root))


if __name__ == '__main__':
    # 对处理的数据取一些 出来可视化
    voc_ds = VOC_Dataset()
    i = 1
    for data, label in voc_ds:
        save_image(data, f"E:\\xydataset\\VOC2012\\Unet_train_voc\\{i}.jpg", nrow=1)
        save_image(label, f"E:\\xydataset\\VOC2012\\Unet_train_voc\\{i}.png", nrow=1)
        i += 1
        if i == 1500:
            exit()
    voc_ds = my_vocdata()
