# -*- coding: utf-8 -*-
# @Time    : 2023/9/14 10:05
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : get_txt.py
# ------❤️❤️❤️------ #

import json
import os


class Get_Txt:
    def __init__(self, data_root_path, train_or_val):
        # self.CLS = ['None', 'missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
        self.data_root_path = data_root_path
        self.label_folder_path = f"{data_root_path}\\labels"
        self.train_or_val = train_or_val
        for i in train_or_val:
            self.json_path = f"{data_root_path}\\Annotations\\{i}.json"
            with open(self.json_path, 'r') as file:
                json_data = json.load(file)
            for key, _ in json_data.items():
                self.images = json_data['images']
                self.annotations = json_data['annotations']
                # self.categories = json_data['categories']
            self.gen_label_txt()
            self.gen_train_and_val_txt(i)

    def gen_label_txt(self):
        if not os.path.exists(self.label_folder_path):
            os.mkdir(self.label_folder_path)
        for i in self.images:
            file_name, height, width, ids = i['file_name'][:-4], i['height'], i['width'], i['id']
            with open(f"{self.label_folder_path}\\{file_name}.txt", 'w') as f:
                for j in self.annotations:
                    if j['image_id'] == ids:
                        x, y, w, h = self.convert((width, height), j['bbox'])
                        cls = j['category_id']
                        f.write(f'{cls} {x} {y} {w} {h} \n')

    def gen_train_and_val_txt(self, i):
        with open(f"{self.data_root_path}\\{i}.txt", 'w') as f:
            for i in self.images:
                file_name = i['file_name']
                f.write(f'{self.data_root_path}\\images\\{file_name}\n')

    def convert(self, size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
        dw = 1. / size[0]  # 1/w
        dh = 1. / size[1]  # 1/h
        x = (box[0] + box[2] / 2)  # 物体在图中的中心点x坐标
        y = (box[1] + box[3] / 2)  # 物体在图中的中心点y坐标
        w = box[2]  # 物体实际像素宽度
        h = box[3]  # 物体实际像素高度
        x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
        w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
        y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
        h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
        x = round(x, 6)
        w = round(w, 6)
        y = round(y, 6)
        h = round(h, 6)
        return x, y, w, h  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


txt_generator = Get_Txt(r'F:\PCB_DATASET', ['train', 'val'])
