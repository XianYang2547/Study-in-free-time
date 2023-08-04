from torchvision import models
from torch import nn
from my_utils import *

#########################################################################
# 加载yolov5权重文件
model = torch.hub.load(r'F:\XY_mts\yolov5-bone_age', 'custom', path=r'F:\XY_mts\yolov5-bone_age\my_bone_age\my_mode\best1.pt',
                       source='local')
model.eval()
model.conf = 0.6  # 置信度

arthrosis = {'MCPFirst': ['MCPFirst', 11],  # 第一手指掌骨
             'DIPFirst': ['DIPFirst', 11],  # 第一手指远节指骨
             'PIPFirst': ['PIPFirst', 12],  # 第一手指近节指骨
             'MIP': ['MIP', 12],  # 中节指骨（除了拇指剩下四只手指）（第一手指【拇指】是没有中节指骨的））
             'Radius': ['Radius', 14],  # 桡骨
             'Ulna': ['Ulna', 12],  # 尺骨
             'PIP': ['PIP', 12],  # 近节指骨（除了拇指剩下四只手指）
             'DIP': ['DIP', 11],  # 远节指骨（除了拇指剩下四只手指）
             'MCP': ['MCP', 10]}  # 掌骨（除了拇指剩下四只手指）

models_ = {}
# 加载九个小模型
paths="F:\XY_mts\yolov5-bone_age\my_bone_age\my_mode"
for name in os.listdir(paths):
    if name.endswith(".pth"):
        bone_name = name.split('_')[0]
        # 创建网络
        # 获取等级数
        leval = arthrosis[bone_name][1]
        m = models.resnet18()
        # 输入通道数  、 输出类别
        # m.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        m.fc = nn.Linear(512, int(leval))
        m.load_state_dict(torch.load(f"{paths}\\{name}"))
        m.eval()
        models_[bone_name] = m
print("模型加载完成")
#########################################################################
path = r'E:\xydataset\bone_age\mydata\images\1527.png'
# 加载图片
img = cv2.imread(path)
# 把图片传入到yolov5中
result = model(path)
boxes = result.xyxy[0]
# 从结果中再得到13根骨节
boxes = get_13_bone(boxes)
# 骨节对应的名称 Radius, Ulna, MCPFirst, MCP, ProximalPhalanx, MiddlePhalanx, DistalPhalanx
aa = ["DIP", 'DIP', 'DIPFirst', "MIP", 'MIP', "PIP", 'PIP', 'PIPFirst', "MCP", 'MCP', "MCPFirst", "Ulna", "Radius"]
bb = ["DIPFifth", 'DIPThird', 'DIPFirst', "MIPFifth", 'MIPThird', "PIPFifth", 'PIPThird', 'PIPFirst', "MCPFifth",
      'MCPThird', "MCPFirst", "Ulna", "Radius"]
#########################################################################

sum_score = 0
sex='boy'
dic = {}
for i in range(len(aa)):
    # 获取左上角右下角坐标
    x1 = int(boxes[i][0])
    y1 = int(boxes[i][1])
    x2 = int(boxes[i][2])
    y2 = int(boxes[i][3])
    # print(x1, y1)
    # 在原图上裁切图片  参数 开始y坐标:结束y坐标 , 开始x坐标:结束x坐标
    img_roi = img[y1:y2, x1:x2]
    if not os.path.exists('img'):
        os.makedirs("img")
    # 保存裁切的图片
    path = f'img\\{i}.png'
    cv2.imwrite(path, img_roi)

    # 把裁剪后的图片进行处理
    s_img = img_pro(path)
    s_img = torch.unsqueeze(s_img, dim=0)
    # 找对应的小模型，传入到小模型
    small_name = aa[i]
    small_mode = models_[small_name]
    leavl = small_mode(s_img)
    # 获取等级索引
    index = int(leavl.argmax(dim=1))
    # print(index)
    # 获取得分
    score = SCORE[sex][bb[i]][index]
    sum_score += score
    dic[bb[i]] = [index + 1, score]

boneAge = calcBoneAge(sum_score, sex)
rr = export(dic, sum_score, boneAge,sex)
print(rr)

# print(boxes)
