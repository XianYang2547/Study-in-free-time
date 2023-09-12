# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 14:46
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : MyData.py
# ------❤️❤️❤️------ #

from torch.utils.data import Dataset
from datasets import load_from_disk

#加载磁盘数据
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


class MyDataset(Dataset):
    def __init__(self, split):
        # 从本地加载数据，train val test模式
        self.dataset = load_from_disk("data/ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """只返回text, label"""
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label

# 可以用于input_sentence_test.py中
if __name__ == '__main__':
    dataset0 = MyDataset("test")
    for data0 in dataset0:
        print(data0)  # ('已经贴完了，又给小区的妈妈买了一套。最值得推荐', 1)

