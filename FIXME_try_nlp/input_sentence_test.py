# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 14:46
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : input_sentence_test.py
# ------❤️❤️❤️------ #

import torch
from transformers import BertTokenizer

from net import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token = BertTokenizer.from_pretrained("model/vocab.txt")
names = ["负向评价", "正向评价"]
model = Model().to(DEVICE)


def collate_fn(data):
    sentes = []
    sentes.append(data)
    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentes,
        truncation=True,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    return input_ids, attention_mask, token_type_ids


def test():
    model.load_state_dict(torch.load("params/12bert.pth"))
    model.eval()
    while True:
        data = input("请输入测试数据（输入'q'退出）：")
        if data == 'q':
            print("测试结束")
            break
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), \
            token_type_ids.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定：", names[out], "\n")


if __name__ == '__main__':
    test()

