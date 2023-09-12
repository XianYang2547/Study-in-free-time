# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 15:11
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : 6.py
# ------❤️❤️❤️------ #
import random

from datasets import load_dataset

# 加载csv格式数据
csv_data = load_dataset(path="csv", data_files=r"data/online_shopping_10_cats.csv")
# print(csv_data)

data = csv_data["train"]
print(data)
# #查看数据
print(len(data))
total_data = len(data)
train_ratio = 0.7  # 训练集比例
val_ratio = 0.15  # 验证集比例
test_ratio = 0.15  # 测试集比例
train_size = int(total_data * train_ratio)
val_size = int(total_data * val_ratio)
test_size = total_data - train_size - val_size
data_indices = list(range(total_data))
random.shuffle(data_indices)
train_data = data_indices[:train_size]
val_data = data_indices[train_size:train_size+val_size]
test_data = data_indices[train_size+val_size:]
print(val_data)
print(val_data[1])
print(data[val_data[1]])
