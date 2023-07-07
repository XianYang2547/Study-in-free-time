import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from datas_loader import Celeba_Dataset
from tqdm import tqdm


# 创建训练器
class Trainer:
    # 网络，参数保存路径，训练数据路径，cuda加速为True
    def __init__(self, net, save_path, dataset_path):
        self.net = net.cuda()
        self.save_path = save_path
        self.dataset_path = dataset_path
        # 置信度损失
        # nn.BCELoss()：二分类交叉熵损失函数，使用之前用sigmoid()激活
        self.conf_loss_fc = nn.BCELoss()
        # 偏移量损失
        self.offset_loss_fc = nn.MSELoss()
        self.landmark_loss_fc = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
            print(f"加载预训练模型{self.save_path}")

    # 训练
    def train(self):
        faceDataset = Celeba_Dataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=256, shuffle=True)
        for epoch in range(1, 100):
            for i, (img, conf, offset, landmark) in tqdm(enumerate(dataloader), total=len(dataloader)):
                # for i, (img, conf, offset) in tqdm(enumerate(dataloader), total=len(dataloader)):
                # 样本输出，置信度，偏移量
                img, conf, offset, landmark = img.cuda(), conf.cuda(), offset.cuda(), landmark.cuda()
                # img, conf, offset = img.cuda(), conf.cuda(), offset.cuda()
                # 输出置信度，偏移量
                out_conf, out_offset, out_landmark = self.net(img)
                out_conf = out_conf.reshape(-1, 1)
                out_offset = out_offset.reshape(-1, 4)
                out_landmark = out_landmark.reshape(-1, 10)
                # 求置信度损失
                # 置信度掩码：求置信度只考虑正、负样本，不考虑部分样本。 逐元素比较两个张量的大小，返回一个布尔类型的张量，指示每个元素是否小于第二个张量对应位置的元素。
                conf_mask = torch.lt(conf, 2)
                # 标签：根据置信度掩码，筛选出置信度为0、1的正、负样本。
                conf_mask_value = torch.masked_select(conf, conf_mask)
                # 网络输出值：预测的“标签”进掩码，返回符合条件的结果
                out_conf_value = torch.masked_select(out_conf, conf_mask)
                # 对置信度做损失
                conf_loss = self.conf_loss_fc(out_conf_value, conf_mask_value)

                # 求偏移量损失：不考虑负样本，只考虑正、部分样本
                offset_mask = torch.gt(conf, 0.)
                # 对置信度大于0的标签(正样本1和部分样本2)，进行掩码；负样本不参与计算，负样本没偏移量
                offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引
                offset = offset[offset_index]  # 标签里偏移量
                out_offset = out_offset[offset_index]  # 输出的偏移量
                offset_loss = self.offset_loss_fc(out_offset, offset)  # 偏移量损失

                landmark_mask = torch.gt(conf, 0.)
                landmark_index = torch.nonzero(landmark_mask)[:, 0]
                landmark1 = landmark[landmark_index]
                out_landmark1 = out_landmark[landmark_index]
                landmark_loss = self.landmark_loss_fc(out_landmark1, landmark1)

                # 总损失
                loss = conf_loss + offset_loss + landmark_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"第{epoch}轮的总损失：{loss.item()}，置信度损失：{conf_loss.item()}，偏移量损失：{offset_loss.item()},landmark_loss:{landmark_loss.item()}")
            torch.save(self.net.state_dict(), self.save_path)
