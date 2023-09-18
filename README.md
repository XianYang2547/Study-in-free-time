<p align="left">
  <a href [https://github.com/XianYang2547/Home-Page]">
  <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub"></a>

## 数据集与框架介绍
印刷电路板（PCB）瑕疵数据集：数据下载链接，是一个公共的合成PCB数据集，由北京大学发布，其中包含1386张图像以及6种缺陷（缺失孔，鼠标咬伤，开路，短路，杂散，伪铜），用于检测，分类和配准任务。我们选取了其中适用与检测任务的693张图像，随机选择593张图像作为训练集，100张图像作为验证集.

## 任务详情
旨在使用`yolov8`训练数据<br>
1.[下载数据](https://aistudio.baidu.com/datasetdetail/52914)<br>
2.使用数据<br>
原始目录：
```
.
├─Annotations
│      train.json
│      train_cpu.json
│      val.json
│      val_cpu.json
├─images
│      01_missing_hole_01.jpg
|      ...
```
处理后的目录：
```
.
│  get_txt.py 从Annotations的json文件中解析生成txt等文件
│  pcb.yml    自己创建写入类别、分类数、train val路径
│  train.txt
│  val.txt
├─Annotations
│      train.json
│      train_cpu.json
│      val.json
│      val_cpu.json 
├─images
│      01_missing_hole_01.jpg
├─labels
│      01_missing_hole_01.txt
```
3.安装yolov8环境依赖<br>
4.在目录下执行`yolo detect model= yolov8n.pt data=pcb.yml batch=8 workers=0`
## Reference
- [https://aistudio.baidu.com/datasetdetail/52914](https://aistudio.baidu.com/datasetdetail/52914)
- [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)



