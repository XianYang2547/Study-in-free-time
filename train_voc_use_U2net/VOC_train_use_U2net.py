import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from MyU2Net import U2NET, U2NETP
from VOC_Dataset import my_vocdata
from tqdm import tqdm

"istrain为ture，加载测试集开启测试，为false开启训练"
istest = False
if not istest:
    # module = r"params/unet_voc.pth"
    img_save_path = r"voc_train_show"
    u2net = U2NET().cuda()
    # opt = torch.optim.Adam(u2net.parameters(), lr=0.01)
    opt = torch.optim.SGD(u2net.parameters(), lr=0.01, momentum=0.9)
    loss = nn.BCELoss()
    loader = DataLoader(my_vocdata(), batch_size=1, shuffle=True)
    # 预训练权重
    if os.path.exists(r'F:\xiany\ZS_Training And Learning\08_seg_VOC_dataset\train_voc_use_U2net\params'):
        u2net.load_state_dict(torch.load(f'params\\u2netp.pth'))
        print("加载unet预训练权重")
    else:
        pass
    for epoch in range(1, 100):
        for i, (img, label) in tqdm(enumerate(loader), total=len(loader)):
            img, label = img.cuda(), label.cuda()
            # u2net的输出有7个，d0是最终的一个
            d0, d1, d2, d3, d4, d5, d6 = u2net(img)
            loss0 = loss(d0, label)
            loss1 = loss(d1, label)
            loss2 = loss(d2, label)
            loss3 = loss(d3, label)
            loss4 = loss(d4, label)
            loss5 = loss(d5, label)
            loss6 = loss(d6, label)
            loss_sum = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            opt.zero_grad()
            loss_sum.backward()
            opt.step()
            # 可视化
            # 1轮2913张
            if i % 50 == 0:
                print(f"第{epoch}-{i}损失{loss_sum.item()}")
                # 写数据：数据、标签、训练结果  cat、stack
                img_ = img[0]
                label_ = label[0]
                out_ = d0[0]
                # 堆叠：CHW
                # img = torch.stack((img_, label_, out_), 0)
                save_image(img_.cpu(), f"{img_save_path}/{epoch}-{i}.tif")
                save_image(label_.cpu(), f"{img_save_path}/{epoch}-{i}.png")
                save_image(out_.cpu(), f"{img_save_path}/{epoch}-{i}.jpg")
        torch.save(u2net.state_dict(), f"params\\{epoch}.pt")
        print("写权重文件成功")
else:
    u2net = U2NET().cuda()
    testloader = DataLoader(my_vocdata(istest=True), batch_size=1, shuffle=True)
    # 预训练权重
    if os.path.exists(r'F:\xiany\ZS_Training And Learning\08_seg_VOC_dataset\train_voc_use_U2net\params'):
        u2net.load_state_dict(torch.load(f'params\\99.pt'))
        print("加载unet预训练权重")
    with torch.no_grad():
        for i, images in tqdm(enumerate(testloader), total=len(testloader)):
            img = images.cuda()
            d0, d1, d2, d3, d4, d5, d6 = u2net(img)
            save_image(d0, f"testresoult/{i}.png")
            save_image(img, f"testresoult/{i}.jpg")
