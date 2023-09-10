# -*- coding: utf-8 -*-
# @Time    : 2023/6/25 18:30
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : rename.py
# @SoftWare：PyCharm


from PIL import Image
import os


def rename_images(directory):
    # 获取目录下的所有文件名
    filenames = os.listdir(directory)
    # 排序，按名字来
    filenames.sort(key=lambda x: int(x[0:-6]))
    i = 0
    for filename in filenames:
        print(filename)
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 构造旧文件的完整路径
            old_path = os.path.join(directory, filename)
            # 打开图像
            image = Image.open(old_path)
            # 构造新的文件名
            new_filename = f"{i}.png"
            # 构造新文件的完整路径
            new_path = os.path.join(r"E:\xydataset\chipdata\train\00", new_filename)
            # 重命名并保存图像
            image.save(new_path)
            # 关闭图像
            image.close()
            # 删除原始文件
            # os.remove(old_path)
        i += 1


# 调用示例
directory = r'E:\xydataset\chipdata\train\label'  # 替换为图像所在的目录路径
rename_images(directory)
