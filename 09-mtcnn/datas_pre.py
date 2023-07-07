# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 19:55
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : datas_pre.py
# @SoftWare：PyCharm


import os
from os.path import join as osp
from PIL import Image
import numpy as np
import utils

img_src = r"E:\xydataset\CelebA\sampledata"
anno_src = r"E:\xydataset\CelebA\sampledata\trainImageList.txt"
save_path = r"E:\xydataset\CelebA\sampledata\outimg"  # 样本保存路径（正样本positive：负样本negative：部分样本part 比例1：3：2）

for face_size in [12, 24, 48]:
    print(f"生成{face_size}的数据")
    # 样本路径：正样本positive、负样本negative、部分样本part
    # 文件夹 -- 文本文件.txt
    positive_image_dir = osp(save_path, str(face_size), "positive")
    negative_image_dir = osp(save_path, str(face_size), "negative")
    part_image_dir = osp(save_path, str(face_size), "part")

    # 自动创建
    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 创建标签文件
    positive_anno_filename = osp(save_path, str(face_size), "positive.txt")
    negative_anno_filename = osp(save_path, str(face_size), "negative.txt")
    part_anno_filename = osp(save_path, str(face_size), "part.txt")

    #  正样本positive:负样本negative:部分样本part = 1: 3: 1
    # 计数器
    positive_count = 0
    negative_count = 0
    part_count = 0

    lines = open(anno_src, "r").readlines()

    # 写文档、抠样本
    positive_anno_file = open(positive_anno_filename, "w")
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")

    try:

        m = 0
        for i in range(len(lines)):

            try:

                strs = lines[i].split()
                image_filename = strs[0].strip()
                image_filename = image_filename.split('\\')

                src_img_path = image_filename[0]
                src_img_name = image_filename[1]

                if m % 1000 == 0:
                    print(str(face_size), "处理图片", m, src_img_path, src_img_name)
                m = m + 1

                # 图片完整路径
                image_file = osp(img_src, src_img_path, src_img_name)
                # 抠图
                with Image.open(image_file) as img:
                    img_w, img_h = img.size

                    # 宽度或高度小于裁剪框12/24/48，无法根据设定值偏离中心点裁剪图片
                    if face_size >= int(min(img_w, img_h) / 2):
                        print("side_len = np.random.randint(face_size, min(img_w, img_h) / 2) ValueError: low >= high",
                              image_file)
                        continue

                    # 建议框的读取
                    sgst_x1 = float(strs[1])
                    sgst_x2 = float(strs[2])
                    sgst_y1 = float(strs[3])
                    sgst_y2 = float(strs[4])
                    # 关键点的读取
                    sgst_px1 = float(strs[5])
                    sgst_py1 = float(strs[6])
                    sgst_px2 = float(strs[7])
                    sgst_py2 = float(strs[8])
                    sgst_px3 = float(strs[9])
                    sgst_py3 = float(strs[10])
                    sgst_px4 = float(strs[11])
                    sgst_py4 = float(strs[12])
                    sgst_px5 = float(strs[13])
                    sgst_py5 = float(strs[14])
                    # 半张脸
                    if sgst_x1 < 0 or sgst_y1 < 0 or (sgst_x2 - sgst_x1) < 0 or (sgst_y2 - sgst_y1) < 0 or max(
                        (sgst_x2 - sgst_x1), (sgst_y2 - sgst_y1)) < 48:
                        continue
                    # 坐标
                    boxes = [[sgst_x1, sgst_y1, sgst_x2, sgst_y2]]

                    # 建议框的中心点坐标
                    # 建议框：位置(中心点2、左上角右下角坐标4)、形状（4，2）
                    # （sgst_x1,sgst_y1,sgst_x2,sgst_y2）（cx1,cy1,width,height）
                    sgst_center_x = int(sgst_x1 + (sgst_x2 - sgst_x1) / 2)
                    sgst_center_y = int(sgst_y1 + (sgst_y2 - sgst_y1) / 2)

                    # 建议框宽、高
                    sgst_w = sgst_x2 - sgst_x1
                    sgst_h = sgst_y2 - sgst_y1

                    # 随机位置，抠样本
                    for _ in range(10):

                        # 中心点随机移动范围 -0.2 , +0.2
                        move_w = np.random.randint(-sgst_w * 0.2, sgst_w * 0.2 + 1)
                        move_h = np.random.randint(-sgst_h * 0.2, sgst_h * 0.2 + 1)
                        # 随机之后的中心点
                        move_center_x = sgst_center_x + move_w
                        move_center_y = sgst_center_y + move_h
                        # 抠图为正方形
                        side_len = np.random.randint(int(min(sgst_w, sgst_h) * 0.5), int(max(sgst_w, sgst_h) * 1.1))
                        if side_len <= 0:
                            continue

                        # 偏移量
                        x1_ = np.max(move_center_x - side_len / 2, 0)
                        y1_ = np.max(move_center_y - side_len / 2, 0)

                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])  # 裁剪框的两个坐标

                        # 偏移量
                        offset_x1 = (sgst_x1 - x1_) / side_len  # 建议框坐标与裁剪坐标  偏移百分比
                        offset_y1 = (sgst_y1 - y1_) / side_len
                        offset_x2 = (sgst_x2 - x2_) / side_len
                        offset_y2 = (sgst_y2 - y2_) / side_len

                        # 五官关键点的偏移量
                        offset_px1 = (sgst_px1 - x1_) / side_len
                        offset_py1 = (sgst_py1 - y1_) / side_len
                        offset_px2 = (sgst_px2 - x1_) / side_len
                        offset_py2 = (sgst_py2 - y1_) / side_len
                        offset_px3 = (sgst_px3 - x1_) / side_len
                        offset_py3 = (sgst_py3 - y1_) / side_len
                        offset_px4 = (sgst_px4 - x1_) / side_len
                        offset_py4 = (sgst_py4 - y1_) / side_len
                        offset_px5 = (sgst_px5 - x1_) / side_len
                        offset_py5 = (sgst_py5 - y1_) / side_len

                        # 抠图
                        face_crop = img.crop(crop_box)  # crop_box 坐标为正方形
                        face_resize = face_crop.resize((face_size, face_size))
                        # 计算iou，判断样本类型
                        iou = utils.IOU(crop_box, np.array(boxes))[0]  # 正方形裁剪框与矩形建议框计算IOU
                        if iou > 0.65:  # 正样本
                            # 写文件
                            # 文件名，置信度：1，建议框坐标的偏移量，关键点坐标的偏移量
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2,
                                    offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4,
                                    offset_py4, offset_px5,
                                    offset_py5
                                ))
                            positive_anno_file.flush()
                            # 保存抠图  celaba/celeba_output/12/positive
                            face_resize.save(osp(positive_image_dir,
                                                 f"{positive_count}.jpg"))  # f"{positive_image_dir}/{positive_count}.jpg")

                            positive_count += 1

                        elif iou > 0.45:
                            # part样本置信度：2
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2,
                                    offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4,
                                    offset_py4, offset_px5, offset_py5
                                ))
                            part_anno_file.flush()
                            # 保存抠图  celaba/celeba_output/12/positive
                            face_resize.save(osp(part_image_dir, f"{part_count}.jpg"))
                            part_count += 1
                        elif iou < 0.3:
                            # 负样本置信度：0
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                    negative_count, 0, offset_x1, offset_y1, offset_x2, offset_y2,
                                    offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4,
                                    offset_py4, offset_px5, offset_py5
                                ))
                            negative_anno_file.flush()
                            # 保存抠图  celaba/celeba_output/12/positive
                            face_resize.save(osp(negative_image_dir, f"{negative_count}.jpg"))
                            negative_count += 1

                    # 负样本不足
                    for i in range(10):
                        # np.random.randint(int(min(sgst_w, sgst_h) * 0.5), int(max(sgst_w, sgst_h) * 1.1))

                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        if side_len <= 0:
                            continue

                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])
                        iou = utils.IOU(crop_box, np.array(boxes))[0]
                        if iou < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size))
                            # 写文件
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                    negative_count, 0, offset_x1, offset_y1, offset_x2, offset_y2,
                                    offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4,
                                    offset_py4, offset_px5, offset_py5
                                ))
                            negative_anno_file.flush()
                            # 保存抠图  celaba/celeba_output/12/positive
                            face_resize.save(osp(negative_image_dir, f"{negative_count}.jpg"))
                            negative_count += 1


            except Exception as e:
                print(e, src_img_path, src_img_name)
                raise e

    finally:
        # 关闭
        positive_anno_file.close()
        part_anno_file.close()
        negative_anno_file.close()
