# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 14:46
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : calculate_acc.py
# ------❤️❤️❤️------ #

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from MyData import MyDataset
from net import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token = BertTokenizer.from_pretrained("model/vocab.txt")

def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids,attention_mask,token_type_ids,labels

#创建数据集
test_dataset = MyDataset("test")
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    #开始测试
    acc = 0
    total = 0
    print(DEVICE)
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("params/12bert.pth"))
    model.eval()

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), \
            token_type_ids.to(DEVICE), labels.to(DEVICE)
        out = model(input_ids, attention_mask, token_type_ids)
        out = out.argmax(dim=1)
        acc +=(out==labels).sum().item()
        total+=len(labels)
    print(acc/total)
    # 0.91