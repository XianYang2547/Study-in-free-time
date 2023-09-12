# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 14:46
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : net.py
# ------❤️❤️❤️------ #

import torch
from transformers import BertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BertModel 用于加载和使用预训练的BERT模型。模型名字：bert-base-chinese
pretrained = BertModel.from_pretrained("model/bert_base-chinese").to(DEVICE)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2) # 只有正向和负向

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out
