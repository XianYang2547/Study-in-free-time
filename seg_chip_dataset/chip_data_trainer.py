# -*- coding: utf-8 -*-
# @Time    : 2023/6/23 17:44
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : chip_data_trainer.py
# @SoftWare：PyCharm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from My_get_chip_data import MyDataset
from MyUnet import MyUnet


class Trainer_For_Chipdata(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = MyUnet().to(self.device)
        self.data = MyDataset()
        self.traindata = DataLoader(self.data, batch_size=1, shuffle=True)
        self.test_data = MyDataset(isTrain=False)
        self.testdata = DataLoader(self.test_data, batch_size=2, shuffle=True)
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.05)

    def start_train(self):
        for epoch in range(21):
            print(f'第{epoch + 1}轮训练开始------->')
            for i, (img, label) in tqdm(enumerate(self.traindata, start=1), total=len(self.traindata)):
                img1 = img
                img, label = img.to(self.device), label.to(self.device)
                out = self.net(img)
                # print(out.shape)
                # print(label.shape)
                # exit()
                loss_fn = self.loss_fn(out, label)
                self.opt.zero_grad()
                loss_fn.backward()
                self.opt.step()
                if i % 20 == 0:
                    save_image(img1, f"chip_train_show/{epoch + 1}_{i}.jpg", nrow=1)
                    save_image(out, f"chip_train_show/{epoch + 1}_{i}.png", nrow=1)
            print(f"损失为{loss_fn}")
            if epoch % 2 == 0:
                torch.save(self.net.state_dict(), f"params\\{epoch}.pt")

    def start_test(self):
        with torch.no_grad():
            self.net.load_state_dict(torch.load(f"params\\20.pt"))
            for i, img_and_filename in tqdm(enumerate(self.testdata), total=len(self.testdata)):
                img, filename = img_and_filename[0], img_and_filename[1]
                img = img.to(self.device)
                test_out = self.net(img)  # torch.Size([2, 3, 512, 512])
                save_image(test_out[0], f"chip_resoult/{filename}.png", nrow=1)
                save_image(img, f"chip_resoult/{filename}.jpg", nrow=1)


if __name__ == '__main__':
    out = Trainer_For_Chipdata()
    # out.start_train()
    out.start_test()
