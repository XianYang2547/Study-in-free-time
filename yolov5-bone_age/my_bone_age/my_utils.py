#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/31 19:45
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : my_utils.py
import math
import os
import cv2
import numpy as np
import torch
import random
import torchvision
from PIL import Image


#########################################################

def adaptive_histogram_equalization(image_path, clip_limit=3.0, tile_grid_size=(8, 8)):
    "对原始数据进行自适应直方图均衡化"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # 对图像进行自适应直方图均衡化
    equalized_image = clahe.apply(image)
    # 返回均衡化后的图像
    return equalized_image


## 大数据集处理
# path=r"E:\xydataset\bone_age\data\VOCdevkit\VOC2007\JPEGImages"
def save_big(path=r'E:\xydataset\bone_age\data\VOCdevkit\VOC2007\JPEGImages', out=r''):
    for image_name in os.listdir(path):
        output_image = adaptive_histogram_equalization(os.path.join(path, image_name))
        cv2.imwrite(f"E:\\xydataset\\bone_age\\data\\VOCdevkit\\VOC2007\\images\\{image_name}", output_image)


##小数据集处理
# path1=r"E:\xydataset\bone_age\orgarthrosis"
# out_save_path=r'E:\xydataset\bone_age\arthrosis'
def save_smiall(path1=r"E:\xydataset\bone_age\orgarthrosis", out_save_path=r'E:\xydataset\bone_age\arthrosis'):
    for big_class in os.listdir(path1):
        path2 = os.path.join(path1, big_class)
        for small_class in os.listdir(path2):  # small_class1-11
            path3 = os.path.join(path2, small_class)
            for img in os.listdir(path3):
                outing = adaptive_histogram_equalization(os.path.join(path3, img))
                savePath = f"{out_save_path}\\{big_class}\\{small_class}"
                if not os.path.exists(savePath): os.makedirs(savePath)
                cv2.imwrite(f"{savePath}\\{img}", outing)


#######################################################
def image_totate(img_path):
    """旋转"""
    img = Image.open(img_path)
    for i in range(1, 6):
        rot = random.randint(-45, 45)
        dst = img.rotate(rot)
        folename, _ = img_path.split('.')
        dst.save(f"{folename}_{i}.png")


def gen_(path=r"E:\xydataset\bone_age\arthrosis"):
    ## 生成图像文件
    for big_class in os.listdir(path):
        print(big_class)
        path0 = os.path.join(path, big_class)  # 'E:\\xydataset\\bone_age\\arthrosis\\DIP'
        for small_class in os.listdir(path0):  # small_class1-11
            print(small_class)
            path1 = os.path.join(path0, small_class)  # 'E:\\xydataset\\bone_age\\arthrosis\\DIP\\1'
            for img in os.listdir(path1):
                image_totate(f"{path1}\\{img}")


#######################################################
def save_file(list, path, name):
    myfile = os.path.join(path, name)  # 全路径
    if os.path.exists(myfile):  # 移除myfile文件
        os.remove(myfile)
    with open(myfile, "w") as f:
        f.writelines(list)  # list写入myfile  将一个字符串列表写入文件中，每个字符串作为一行写入


def huafen(path=r"E:\xydataset\bone_age\arthrosis", train_persent=0.9):
    for big_class in os.listdir(path):
        path0 = os.path.join(path, big_class)  # 'E:\\xydataset\\bone_age\\arthrosis\\DIP'
        train_list = []
        val_list = []
        for small_class in os.listdir(path0):  # small_class1-11
            path1 = os.path.join(path0, small_class)  # 'E:\\xydataset\\bone_age\\arthrosis\\DIP\\1'

            num = len(os.listdir(path1))
            train_num = num * train_persent

            for index, img in enumerate(os.listdir(path1), start=1):
                if index < train_num:
                    train_list.append(f"{path1}\\{img} {small_class}\n")
                    random.shuffle(train_list)
                else:
                    val_list.append(f"{path1}\\{img} {small_class}\n")
                    random.shuffle(val_list)
        random.shuffle(train_list)  # 打乱顺序
        random.shuffle(val_list)
        save_file(train_list, path0, 'train.txt')
        save_file(val_list, path0, 'val.txt')


#######################################################
def count_files_in_directory(directory=r"E:\xydataset\bone_age\arthrosis"):
    ## 统计文件个数
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    print(f"当前目录下的文件个数为: {count}")


######################################################
# 用于筛选关节
# 从detect.py第139行，从21个输出中筛选出所需的3个

# a=torch.tensor(
#        [[1.42969e+02, 1.49508e+02, 1.97098e+02, 1.86749e+02, 7.68790e-01, 0.00000e+00],
#         [1.93550e+02, 1.25177e+02, 2.43234e+02, 1.59354e+02, 7.56057e-01, 1.00000e+00],
#         [3.35595e+02, 2.68068e+02, 3.86490e+02, 3.14087e+02, 7.45742e-01, 2.00000e+00],
#         [2.09110e+02, 1.74374e+02, 2.54028e+02, 2.26726e+02, 7.45573e-01, 3.00000e+00],
#         [2.70209e+02, 6.68984e+01, 3.10977e+02, 1.12979e+02, 7.44269e-01, 3.00000e+00],
#         [2.63559e+02, 1.24737e+02, 3.16228e+02, 1.62179e+02, 7.42185e-01, 3.00000e+00],
#         [2.60193e+02, 2.12233e+02, 2.98518e+02, 2.78922e+02, 7.37635e-01, 3.00000e+00],
#         [3.76215e+02, 2.34105e+02, 4.44949e+02, 2.83756e+02, 7.36909e-01, 4.00000e+00],
#         [2.59769e+02, 1.72711e+02, 3.05187e+02, 2.23363e+02, 7.32885e-01, 4.00000e+00],
#         [7.01941e+01, 1.69894e+02, 1.11127e+02, 2.07482e+02, 7.22949e-01, 4.00000e+00],
#         [1.89925e+02, 2.36282e+02, 2.22445e+02, 2.93318e+02, 7.20568e-01, 4.00000e+00],
#         [1.80039e+02, 5.55789e+01, 2.25824e+02, 1.09349e+02, 7.15672e-01, 4.00000e+00],
#         [2.19087e+02, 2.19804e+02, 2.57147e+02, 2.80347e+02, 7.02796e-01, 5.00000e+00],
#         [2.76868e+02, 3.04695e+02, 3.44617e+02, 3.66541e+02, 7.02271e-01, 5.00000e+00],
#         [1.04326e+02, 2.06106e+02, 1.42857e+02, 2.34103e+02, 6.91653e-01, 5.00000e+00],
#         [1.29374e+02, 8.84383e+01, 1.67659e+02, 1.36347e+02, 6.86103e-01, 5.00000e+00],
#         [1.63151e+02, 2.57619e+02, 1.95838e+02, 3.12427e+02, 6.84272e-01, 6.00000e+00],
#         [1.74399e+02, 1.96544e+02, 2.12738e+02, 2.45343e+02, 6.82753e-01, 6.00000e+00],
#         [1.36918e+02, 2.32864e+02, 1.76699e+02, 2.73000e+02, 6.81680e-01, 6.00000e+00],
#         [2.30641e+02, 3.69652e+02, 2.94179e+02, 4.60157e+02, 5.70022e-01, 6.00000e+00],
#         [1.90784e+02, 3.71902e+02, 2.40338e+02, 4.64401e+02, 3.80538e-01, 6.00000e+00]])
def get_(det, clazz, index):
    ind = torch.where(det[:, 5] == clazz)
    b = det[ind]
    bb = b[:, 0].argsort()
    bbb = b[bb]
    res = bbb[index]
    return res

def get_13_bone(det):
    if len(det)==21:
        Radius = get_(det, 0, [0])
        Ulna = get_(det, 1, [0])
        MCPFirst = get_(det, 2, [0])
        MCP = get_(det, 3, [0, 2])
        ProximalPhalanx = get_(det, 4, [0, 2, 4])
        MiddlePhalanx = get_(det, 5, [0, 2])
        DistalPhalanx = get_(det, 6, [0, 2, 4])

        res = torch.cat([Radius, Ulna, MCPFirst, MCP, ProximalPhalanx, MiddlePhalanx, DistalPhalanx], dim=0)
        return res
    else:
        print('骨节数量不是21')
######################################################
# mydataset里面用的函数
def one_hot( num, i):
    b, i = np.zeros(num), int(i) - 1  # 编一个类别数的一维o数组
    b[i] = 1.  # 在指定的位置填充1
    return b

def img_pro(path):
    img = Image.open(path)  # 打开图片
    # print(savepath)
    w, h = img.size[0], img.size[1]  # 获取图片的宽高
    a = np.maximum(w, h)  # 获取wh中的最大值
    goal_img = Image.new('RGB', (a, a), color=(128, 128, 128))  # 创造一个黑板图片
    goal_img.paste(img, (int((a - w) / 2), int((a - h) / 2)))  # 将图片贴到目标图片上
    image = goal_img.resize((224, 224))  # 改变图片大小
    # image.show()
    image = torchvision.transforms.ToTensor()(image)
    return image

######################################################
# bone_age 计算结果用的
def calcBoneAge(score, sex):
    # 根据总分计算对应的年龄
    if sex == 'boy':
        boneAge = 2.01790023656577 + (-0.0931820870747269) * score + math.pow(score, 2) * 0.00334709095418796 + \
                  math.pow(score, 3) * (-3.32988302362153E-05) + math.pow(score, 4) * (1.75712910819776E-07) + \
                  math.pow(score, 5) * (-5.59998691223273E-10) + math.pow(score, 6) * (1.1296711294933E-12) + \
                  math.pow(score, 7) * (-1.45218037113138e-15) + math.pow(score, 8) * (1.15333377080353e-18) + \
                  math.pow(score, 9) * (-5.15887481551927e-22) + math.pow(score, 10) * (9.94098428102335e-26)
        return round(boneAge, 2)
    elif sex == 'girl':
        boneAge = 5.81191794824917 + (-0.271546561737745) * score + \
                  math.pow(score, 2) * 0.00526301486340724 + math.pow(score, 3) * (-4.37797717401925E-05) + \
                  math.pow(score, 4) * (2.0858722025667E-07) + math.pow(score, 5) * (-6.21879866563429E-10) + \
                  math.pow(score, 6) * (1.19909931745368E-12) + math.pow(score, 7) * (-1.49462900826936E-15) + \
                  math.pow(score, 8) * (1.162435538672E-18) + math.pow(score, 9) * (-5.12713017846218E-22) + \
                  math.pow(score, 10) * (9.78989966891478E-26)
        return round(boneAge, 2)

def export(results, score, boneAge,sex):
    report = """
    you are {}
    第一掌骨骺分级{}级，得{}分；
    第三掌骨骨骺分级{}级，得{}分；
    第五掌骨骨骺分级{}级，得{}分；
    第一近节指骨骨骺分级{}级，得{}分；
    第三近节指骨骨骺分级{}级，得{}分；
    第五近节指骨骨骺分级{}级，得{}分；
    第三中节指骨骨骺分级{}级，得{}分；
    第五中节指骨骨骺分级{}级，得{}分；
    第一远节指骨骨骺分级{}级，得{}分；
    第三远节指骨骨骺分级{}级，得{}分；
    第五远节指骨骨骺分级{}级，得{}分；
    尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

    RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
        sex,
        results['MCPFirst'][0], results['MCPFirst'][1], \
        results['MCPThird'][0], results['MCPThird'][1], \
        results['MCPFifth'][0], results['MCPFifth'][1], \
        results['PIPFirst'][0], results['PIPFirst'][1], \
        results['PIPThird'][0], results['PIPThird'][1], \
        results['PIPFifth'][0], results['PIPFifth'][1], \
        results['MIPThird'][0], results['MIPThird'][1], \
        results['MIPFifth'][0], results['MIPFifth'][1], \
        results['DIPFirst'][0], results['DIPFirst'][1], \
        results['DIPThird'][0], results['DIPThird'][1], \
        results['DIPFifth'][0], results['DIPFifth'][1], \
        results['Ulna'][0], results['Ulna'][1], \
        results['Radius'][0], results['Radius'][1], \
        score, boneAge)
    return report

SCORE = {
    'girl': {
    'Radius': [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
    'Ulna': [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
    'MCPFirst': [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
    'MCPThird': [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
    'MCPFifth': [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
    'PIPFirst': [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
    'PIPThird': [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
    'PIPFifth': [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
    'MIPThird': [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
    'MIPFifth': [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
    'DIPFirst': [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
    'DIPThird': [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
    'DIPFifth': [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49]
},
    'boy': {
        'Radius': [8, 11, 15, 18, 31, 46, 76, 118, 135, 171, 188, 197, 201, 209],
        'Ulna': [25, 30, 35, 43, 61, 80, 116, 157, 168, 180, 187, 194],

        'MCPFirst': [4, 5, 8, 16, 22, 26, 34, 39, 45, 52, 66],
        'MCPThird': [3, 4, 5, 8, 13, 19, 30, 38, 44, 51],
        'MCPFifth': [3, 4, 6, 9, 14, 19, 31, 41, 46, 50],
        'PIPFirst': [4, 5, 7, 11, 17, 23, 29, 36, 44, 52, 59, 66],
        'PIPThird': [3, 4, 5, 8, 14, 19, 23, 28, 34, 40, 45, 50],
        'PIPFifth': [3, 4, 6, 10, 16, 19, 24, 28, 33, 40, 44, 50],
        'MIPThird': [3, 4, 5, 9, 14, 18, 23, 28, 35, 42, 45, 50],
        'MIPFifth': [3, 4, 6, 11, 17, 21, 26, 31, 36, 40, 43, 49],
        'DIPFirst': [4, 5, 6, 9, 19, 28, 36, 43, 46, 51, 67],
        'DIPThird': [3, 4, 5, 9, 15, 23, 29, 33, 37, 40, 49],
        'DIPFifth': [3, 4, 6, 11, 17, 23, 29, 32, 36, 40, 49]
    }
}


















