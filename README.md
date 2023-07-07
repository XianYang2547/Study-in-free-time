<p align="left">
  <a href [https://github.com/XianYang2547/Home-Page]">
  <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub"></a>

# <p align="center">:blush::blush::blush:MTCNN:blush::blush::blush:</p>

                    
![image](img/6.jpg)

# 文件说明
datas_pre.py  生成mtcnn中3个子网络的输入尺寸12*12,24*24,48*48，正、负样本、部分样本以及标签文件\
datas_loader.py 实现自己数据集的类，返回img_data, conf, offset, landmark\
My_net.py 实现p、r、o网络的代码\
train.py 训练脚本\
detects.py 检测脚本\
utils.py 工具类\
