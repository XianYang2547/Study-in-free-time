import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from MyUnet import MyUnet
from VOC_Dataset import VOC_Dataset
from tqdm import tqdm

"可以重构一下，写出函数或者类"
# module = r"params/unet_voc.pth"
img_save_path = r"voc_train_show"
unet = MyUnet().cuda()
opt = torch.optim.Adam(unet.parameters(), lr=0.01)
loss = nn.MSELoss()
loader = DataLoader(VOC_Dataset(), batch_size=1, shuffle=True)
# 预训练权重
# if os.path.exists(module):
#     u2net.load_state_dict(torch.load(module))
#     print("加载unet预训练权重")

for epoch in range(1, 100):
    for i, (img, label) in tqdm(enumerate(loader), total=len(loader)):
        img, label = img.cuda(), label.cuda()
        out = unet(img)
        loss_value = loss(out, label)
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        # 可视化
        # 1轮2913张
        if i % 100 == 0:
            print(f"第{epoch}-{i}损失{loss_value.item()}")
            # 写数据：数据、标签、训练结果  cat、stack
            img_ = img[0]
            label_ = label[0]
            out_ = out[0]
            # 堆叠：CHW 他们维度一样的
            img = torch.stack((img_, label_, out_), 0)
            save_image(img.cpu(), f"{img_save_path}/{epoch}-{i}.jpg")
    torch.save(unet.state_dict(), f"params\\{epoch}.pt")
