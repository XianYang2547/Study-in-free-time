<p align="left">
  <a href [https://github.com/XianYang2547]">
  <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub"></a>


<br>
''结合''<br>
1.安装vs2019，选择使用c++的桌面开发<br>
2.在(pytorch)[https://pytorch.org/]官网下载libtorch-cpu, 据说LibTorch版本和自己转换torchscript模型的torch版本要匹配， (OpenCV)[https://opencv.org/]官网下载opencv4.8.0<br>
3.打开CLion，新建项目，选择c++可执行文件<br>
4.在CLion设置中，工具链使用vs环境，CMake构建类型为Release<br>
5.解压libtorch和opencv，在CMakeLists.txt中修改路径，project name<br>
6.
