<p align="left">
  <a href [https://github.com/XianYang2547]">
  <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub"></a>

<p align="center">记录AX620A使用</p>

## 目录
```
.
├─config
│  └─config.prototxt                         根据官网说明配置
│  └─config_out.prototxt
├─dataset                                    
│  └─test.tar                                数据集
├─gt
├─images
│  └─3799794b483fc2b50a985d167fbfd893.jpeg   几张图像
│  └─....
├─model                                       
   └─best.onnx                               （yolov8）训练后转换的onnx文件
   └─ball.lava_joint
   └─ball.joint                              转为AX620A需要的joint模型文件
```

## 流程
<p align="center"> 
<img src="image/1.png">
</p>

## 说明
1. 模型转onnx时需要解cat操作，见[prophet_mu的博客...](https://www.yuque.com/prophetmu/chenmumu/m3axpi)，但修改ultralytics的模型导出文件时，如果使用CLI导出，则需要进入源码修改*使用ultralytics版本为8.0.96及之前的版本*</p>
2. onnx转AX620A所需要的joint模型文件时，有好多好多坑，特别是跟自己电脑cpu有关
## 结果
<p align="center"> 
<img src="image/2.jpg">
</p>

## Reference
- [AXera-Pi爱芯派官方文档](https://wiki.sipeed.com/ai/zh/deploy/ax-pi.html)
- [AXear Pulsar工具链](https://pulsar-docs.readthedocs.io/zh_CN/latest/)
- [prophet_mu的博客...](https://www.yuque.com/prophetmu/chenmumu/m3axpi)



