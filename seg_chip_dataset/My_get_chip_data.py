# -*- coding: utf-8 -*-
# @Time    : 2023/6/23 9:02
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : My_get_chip_data.py
# @SoftWare：PyCharm


from torch.utils.data import Dataset
import torch, os,cv2
from torchvision import transforms
from torchvision.utils import save_image

root = r"E:\xydataset\chipdata"


class MyDataset(Dataset):
    def __init__(self, root=r"E:\xydataset\chipdata", isTrain=True):
        super().__init__()
        self.train_or_test = 'train' if isTrain else 'test'
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.size= (512, 512)
        self.traindata_path = f"{root}\\{'train'}\\{'image'}"
        self.trainlabel_path = f"{root}\\{'train'}\\{'label'}"
        self.test_path = f"{root}\\{'test'}\\{'image'}"
        # self.__getitem__(0)

    def __getitem__(self, item):
        if self.train_or_test == 'train':
            # 构建图像文件路径，取数据
            datas = os.listdir(self.traindata_path)
            labels = os.listdir(self.trainlabel_path)
            data_path = os.path.join(self.traindata_path, datas[item])
            label_path = os.path.join(self.trainlabel_path, labels[item])

            " 检查数据，分割名字"
            # filename = os.path.basename(data_path)
            # splitname = os.path.splitext(filename)
            # filename1 = os.path.basename(label_path)
            # splitname1 = os.path.splitext(filename1)

            # img = Image.open(data_path)  # (3648, 2736)
            # lab = Image.open(label_path)
            # img = img.resize(self.size)
            # lab = lab.resize(self.size)

            images=cv2.imread(data_path)
            images=cv2.resize(images,self.size)
            # images = np.transpose(images, (2, 0, 1))
            # print(images.shape) # (512, 512, 3)
            labs=cv2.imread(label_path)
            labs=cv2.resize(labs,self.size)
            # print(labs.shape)
            # labs=np.transpose(labs,(2,0,1))
            # print(labs.shape) # (512, 512, 3)

            # save_image(self.transform(img), f"chip_resoult/{splitname[0]}.tif", nrow=1)
            # save_image(self.transform(lab), f"chip_resoult/{splitname1[0]}.png", nrow=1)
            return self.transform(images), self.transform(labs)
        else:
            datas = os.listdir(self.test_path)
            data_path = os.path.join(self.test_path, datas[item])
            # 分离出名字，不要后缀
            filename = os.path.basename(data_path)
            splitname = os.path.splitext(filename)

            # img = Image.open(data_path)
            # img = img.resize(self.size)
            img=cv2.imread(data_path)
            img=cv2.resize(img,self.size)
            return self.transform(img), splitname[0]

    def __len__(self):
        return len(os.listdir(self.traindata_path)) if self.train_or_test == 'train' else len(
            os.listdir(self.test_path))


if __name__ == '__main__':
    datas = MyDataset(root, isTrain=True)

    # 调用类，从__getitem__中返回 训练数据和标签图像，进行保存
    # 测试的话返回测试数据图像和名字
    # i = 1
    # for x, y in datas:
    #     save_image(x, f"img/{i}.jpg")
    #     save_image(y, f"img/{i}.png")
    #     i += 1
