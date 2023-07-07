# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 17:30
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : train_p.py
# @SoftWare：PyCharm

from train import Trainer
from My_net import P_Net

trainp = Trainer(P_Net(), save_path=r"params\6.pth", dataset_path=r"E:\xydataset\CelebA\sampledata\outimg\12")
trainp.train()
