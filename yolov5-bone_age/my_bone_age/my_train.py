#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/31 19:45
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : my_train.py

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_dataset import My_data
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

arthrosises = {'MCPFirst': ['MCPFirst', 11],  # 第一手指掌骨
               'DIPFirst': ['DIPFirst', 11],  # 第一手指远节指骨
               'PIPFirst': ['PIPFirst', 12],  # 第一手指近节指骨
               'MIP'     : ['MIP', 12],  # 中节指骨（除了拇指剩下四只手指）（第一手指【拇指】是没有中节指骨的））
               'Radius'  : ['Radius', 14],  # 桡骨
               'Ulna'    : ['Ulna', 12],  # 尺骨
               'PIP'     : ['PIP', 12],  # 近节指骨（除了拇指剩下四只手指）
               'DIP'     : ['DIP', 11],  # 远节指骨（除了拇指剩下四只手指）
               'MCP'     : ['MCP', 10]}  # 掌骨（除了拇指剩下四只手指）

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Train(nn.Module):
    def __init__(self, arthrosis):
        super().__init__()
        self.train_data = DataLoader(My_data(arthrosis, 'train'), 32, shuffle=True)
        self.val_data = DataLoader(My_data(arthrosis, 'val'), arthrosis[1], shuffle=True)
        self.mode_path = f"F:\XY_mts\yolov5-bone_age\my_bone_age\\my_mode\\{arthrosis[0]}_new.pth"
        ## 加载resnet预训练模型，修改分类数
        self.res_18 = models.resnet18(pretrained=True)
        self.res_18.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        num_features = self.res_18.fc.in_features
        num_classes = arthrosis[1]
        self.res_18.fc = nn.Linear(num_features, num_classes)
        self.res_18.to(device)
        self.opt = torch.optim.Adam(self.res_18.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=f"logs\\{arthrosis[0]}_log_file")
        self.my_train(arthrosis)
    def my_train(self, arthrosis):
        # if os.path.exists(self.mode_path):
        #     self.res_18.load_state_dict(torch.load(self.mode_path))
        best_acc = 0
        for epoch in range(1, 50):
            print('\n', f"小分类模型{arthrosis[0]}第{epoch}轮训练测试开始....(总共50轮)")
            # ************************************  训练   ************************************#
            train_loss = 0
            self.res_18.train()
            for i, (img, label) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                img, label = img.to(device), label.to(device)
                out = self.res_18(img)
                loss = self.loss(out, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                train_loss += loss.item()
            train_avg_loss = train_loss / len(self.train_data)
            self.writer.add_scalar(f"{arthrosis[0]}のtrain loss", train_avg_loss, epoch)
            print(f'第{epoch}个epoch的损失为{train_avg_loss}')
            # ************************************  验证   ************************************#
            print('valing...\n')
            self.res_18.eval()
            with torch.no_grad():
                val_loss = 0
                accs = 0
                for i, (img, label) in tqdm(enumerate(self.val_data), total=len(self.val_data)):
                    img, label = img.to(device), label.to(device)
                    out = self.res_18(img)
                    loss = self.loss(out, label)
                    val_loss += loss.item()
                    acc = torch.mean(torch.eq(out.argmax(dim=1), label.argmax(dim=1)).float())
                    accs += acc.item()
                val_avg_loss = val_loss / len(self.val_data)
                val_avg_acc = accs / len(self.val_data)
                print('val_avg_loss  ', val_avg_loss, '\n',
                      'val_avg_acc  ', val_avg_acc)
                if val_avg_acc > best_acc:
                    best_acc = val_avg_acc
                    torch.save(self.res_18.state_dict(), self.mode_path)
                    print('保存模型')
                self.writer.add_scalar(f"{arthrosis[0]}のval acc", val_avg_acc, epoch)


if __name__ == '__main__':
    for key, value in arthrosises.items():
        print(key, value)
        trains = Train(value)
