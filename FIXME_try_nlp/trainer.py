# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 14:46
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : trainer.py
# ------❤️❤️❤️------ #

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW

from MyData import MyDataset
from net import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token = BertTokenizer.from_pretrained("model/vocab.txt")


def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    cats = [i[2] for i in data]
    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=500,  # 500长度
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)
    cats = torch.LongTensor(cats)

    return input_ids, attention_mask, token_type_ids, labels, cats


# 创建数据集 _Mydata中一般返回text, label，不进行其他操作
train_dataset = MyDataset("train")
train_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=True, drop_last=True, collate_fn=collate_fn)

if __name__ == '__main__':
    # 开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        for i, (input_ids, attention_mask, token_type_ids, labels, cats) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels, cats = input_ids.to(DEVICE), attention_mask.to(DEVICE), \
                token_type_ids.to(DEVICE), labels.to(DEVICE), cats.to(DEVICE)
            out, out1 = model(input_ids, attention_mask, token_type_ids)
            loss1 = loss_func(out, labels)
            loss2 = loss_func(out1, cats)

            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(loss1.item(),loss2.item())
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                out1 = out1.argmax(dim=1)
                acc1 = (out1 == cats).sum().item() / len(cats)

                print(epoch, i, loss.item(), acc, acc1)
                torch.save(model.state_dict(), f"params/{epoch}bert.pth")
        print(epoch, "saved...")
