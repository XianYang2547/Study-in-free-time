# -*- coding: utf-8 -*-
# @Time    : 2023/8/23 10:42
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : my_train.py
# ------❤️❤️❤️------ #

import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datas.get_data import My_Data
from my_loss.arc_loss import Arc_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_dir = "logs"
writer = SummaryWriter(log_dir=log_dir)
model_path = 'weight/best.pt'


class My_Face(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = models.densenet161()
        self.arc = Arc_loss(self.layer1.classifier.out_features, 21)  # 21个人

    def forward(self, x):
        features = self.layer1(x)  # N feature1000
        out = self.arc(features)  # N 批次 1 21个人
        return features, out

    def get(self, x):
        return self.layer1(x)


def train_face():
    # ----------------加载网络、损失函数、优化器--------------------#
    net = My_Face().to(device)
    loss_fn = nn.NLLLoss()

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        print(f"load {model_path} done ...")
    else:
        opt = torch.optim.Adam(net.parameters())
    # ----------------加载训练、验证数据--------------------#
    train_data = My_Data(r'datas/train.txt')
    trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_data = My_Data(r'datas/val.txt')
    valloader = DataLoader(val_data, batch_size=16, shuffle=True)
    # ----------------训练 and 验证--------------------#
    best_loss = float('inf')  # 初始化最佳损失为正无穷
    for epoch in range(101):
        net.train()
        total_train_loss = 0
        total_train_sampls = 0
        for train_img, train_name in tqdm(trainloader, total=len(trainloader)):
            train_img, train_name = train_img.to(device), train_name.to(device)
            features, out = net(train_img)
            loss = loss_fn(out, train_name)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_loss += loss.item() * train_img.size(0)
            total_train_sampls += train_img.size(0)
        avg_train_loss = total_train_loss / total_train_sampls
        print(f"{epoch}-🚀🚀🚀-Train Loss--:  {avg_train_loss}")
        writer.add_scalar('Train_Loss', avg_train_loss, epoch)

        # TODO 测试
        net.eval()
        total_val_loss = 0
        total_val_samples = 0
        for val_img, val_name in tqdm(valloader, total=len(valloader)):
            val_img, val_name = val_img.to(device), val_name.to(device)
            with torch.no_grad():
                _, val_out = net(val_img)
            loss0 = loss_fn(val_out, val_name)
            total_val_loss += loss0.item() * val_img.size(0)
            total_val_samples += val_img.size(0)
        avg_val_loss = total_val_loss / total_val_samples
        print(f"{epoch}-🚀🚀🚀-Val Loss--:  {avg_val_loss}" + '\n')
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(net.state_dict(), 'weight/best.pt')
            print('****************** save models ********************')
        writer.add_scalar('Val_Loss', avg_val_loss, epoch)
    # 保存最后训练的模型
    torch.save(net.state_dict(), 'weight/last.pt')
    writer.close()


if __name__ == '__main__':
    train_face()
