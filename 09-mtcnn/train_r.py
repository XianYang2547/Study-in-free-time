# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 17:30
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : train_p.py
# @SoftWare：PyCharm

from train import Trainer
from My_net import R_Net

trainr = Trainer(R_Net(), save_path=r"params\66.pth", dataset_path=r"E:\xydataset\CelebA\sampledata\outimg\24")
trainr.train()
