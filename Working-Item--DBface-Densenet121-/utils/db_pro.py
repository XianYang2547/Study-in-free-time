# -*- coding: utf-8 -*-
# @Time    : 2023/8/18 14:09
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : db_pro.py
# ------❤️❤️❤️------ #

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from datas.get_data import transform
from utils import common
from utils.DBFace import DBFace
from utils.common import new_crops

HAS_CUDA = torch.cuda.is_available()


def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5):
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


# def detect_image(model, file):
#     image = common.imread(file)
#     objs = detect(model, image)
#     coor = []
#     left = []
#     right = []
#     for obj in objs:
#         common.drawbbox(image, obj)
#         # 取出框和眼睛坐标
#         coor.append(common.intv(obj.box))
#         left.append(common.intv(obj.landmark[0]))
#         right.append(common.intv(obj.landmark[1]))
#
#     # 创建保存总目录
#     if os.path.exists("../result/"):
#         common.delete_folder_contents("../result/")
#     if not os.path.exists("../result/"):
#         os.makedirs("../result/")
#
#     dect_img_path = "../result/" + common.file_name_no_suffix(file) + "_dect.tiff"  # 人脸检测图像路径，用于qt显示
#     common.imwrite(dect_img_path, image)  # 开始保存
#     crop_faces, crop_save_path = new_crops(file, coor, left, right)  # 进行裁切人脸，    crop_save_path用于qt显示
#
#     return dect_img_path, crop_faces, crop_save_path,
#     # 将裁切出来的人脸拿去提取特征，crop_faces[0 1 2] 里面包含检测到的脸，可以用cv2显示和保存


# def image_demo(dbface, path):
#     "QT中使用"
#     path0, crop_faces, crop_save_path = detect_image(dbface, path)
#     return path0, crop_faces, crop_save_path

def detect_image(model, frame):
    objs = detect(model, frame)
    coor = []
    left = []
    right = []
    for obj in objs:
        # common_test.drawbbox(frame, obj)#
        # 取出框和眼睛坐标
        coor.append(common.intv(obj.box))
        left.append(common.intv(obj.landmark[0]))
        right.append(common.intv(obj.landmark[1]))

    crop_faces = new_crops(frame, coor, left, right)
    return objs, crop_faces


def image_demos(path):
    dbface = DBFace()
    dbface.eval()
    if HAS_CUDA:
        dbface.cuda()
    dbface.load('MyQT/QT_resources/dbface.pth')
    frame = cv2.imread(path)
    _, crop_faces = detect_image(dbface, frame)
    return crop_faces


def detect_images(model, frame):
    objs = detect(model, frame)
    coor = []
    left = []
    right = []
    for obj in objs:
        # common_test.drawbbox(frame, obj)  #
        # 取出框和眼睛坐标
        coor.append(common.intv(obj.box))
        left.append(common.intv(obj.landmark[0]))
        right.append(common.intv(obj.landmark[1]))

    crop_faces = new_crops(frame, coor, left, right)  # 进行裁切人脸，    crop_save_path用于qt显示

    return objs, crop_faces


def compare(face1, face2):
    "检测的人脸特征和库中特征比对，计算余弦相似度"
    # "cos"  bug1   face1 [1,1000]  face2   [1000]
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    res = torch.matmul(face1_norm, face2_norm.t())
    return res


def compare1(face1, face2_list):
    "传入检测到的人脸，和数据库中存在的某个人的人脸特征列表对比"
    face1_norm = F.normalize(face1)
    face2_list_norm = [F.normalize(face2) for face2 in face2_list]
    res_list = []
    for face2_norm in face2_list_norm:
        res = torch.matmul(face1_norm, face2_norm.to('cpu').t())
        res_list.append(res.item())
    return res_list


def Face_comparison(models, img, face_data, thres):
    '调用摄像头识别'
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i = Image.fromarray(rgb_img)  # 转为pil类型
    with torch.no_grad():
        p1_feature = models.get(transform(i).unsqueeze(0).cuda())

    for key, value in face_data.items():
        res = compare1(p1_feature.to('cpu'), value)
        if sum(res) / len(res) > thres:  # 取平均
            return key, torch.tensor(sum(res) / len(res))
        else:
            continue
    return 'Unknown', 'UnRecognized'


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)))
        self.func = nn.Softmax()

    def forward(self, x, s=1, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        return torch.log(arcsoftmax)


def add_filedoer(name, facelist, net, device, path):
    # 是否存在本地库

    if os.path.exists(path):
        face_fea = torch.load(path)
    else:
        face_fea = {}
    for i in facelist:
        rgb_img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i = Image.fromarray(rgb_img)  # 转为pil类型
        with torch.no_grad():
            p1_feature = net.get(transform(i).unsqueeze(0).to(device))
        if name not in face_fea:
            face_fea[name] = [p1_feature]
        else:
            face_fea[name].append(p1_feature)
    torch.save(face_fea, path)
