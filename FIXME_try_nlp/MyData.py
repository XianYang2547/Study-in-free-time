# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 14:46
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : MyData.py
# ------❤️❤️❤️------ #
import random

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from datasets import load_dataset
import torch.nn.functional as F

# 加载磁盘数据
# dataset = load_from_disk("data/ChnSentiCorp")
# print(dataset) # 查看这个数据集
# """DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 9600
#     })
#     validation: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1200
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1200
#     })
# })"""
# # #取出训练集
# train_data = dataset["train"]
# print(train_data)
# #查看其中一条
# for data in train_data:
#     print(data)
#     exit()
# {'text': '选择珠江花园的原因就是方便，还算丰富。 服务吗，一般','label': 1}

cats = ['书籍', '平板', '手机', '水果', '洗发水', '热水器', '蒙牛', '衣服', '计算机', '酒店']


class MyDataset(Dataset):
    def __init__(self, split):
        # 从本地加载数据，train val test模式
        self.dataset = load_dataset(path="csv", data_files=r"data/online_shopping_10_cats.csv")
        data = self.dataset["train"]
        # 划分
        total_data = len(data)  # 62774
        train_ratio = 0.7  # 训练集比例
        val_ratio = 0.15  # 验证集比例
        test_ratio = 0.15  # 测试集比例
        train_size = int(total_data * train_ratio)  # 43941
        val_size = int(total_data * val_ratio)  # 9416
        test_size = total_data - train_size - val_size  # 9417
        data_indices = list(range(total_data))
        random.shuffle(data_indices)
        train_data = data_indices[:train_size]
        val_data = data_indices[train_size:train_size + val_size]
        test_data = data_indices[train_size + val_size:]

        if split == "train":
            self.dataset = [data[i] for i in train_data]
        elif split == "validation":
            self.dataset = [data[i] for i in val_data]
        elif split == "test":
            self.dataset = [data[i] for i in test_data]
        self.__getitem__(1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """只返回text, label"""
        text = self.dataset[item]["review"]
        label = self.dataset[item]["label"]
        cat = self.dataset[item]["cat"]
        position = cats.index(cat)
        # one_hot_length = len(cats)
        # one_hot = F.one_hot(torch.tensor([position]), one_hot_length).squeeze()
        return text, label, position


# 可以用于input_sentence_test.py中
if __name__ == '__main__':
    dataset0 = MyDataset("test")
    for data0 in dataset0:
        print(data0)  # ('已经贴完了，又给小区的妈妈买了一套。最值得推荐', 1)
