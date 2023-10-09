#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

int main() {
    cv::Mat image = cv::imread("data/2.jpg", 0);//读取图片
    cv::Mat imagesize = cv::Mat(cv::Size(28, 28), image.type());
    cv::resize(image, imagesize, cv::Size(28, 28));
    // 转为tensor
    torch::Tensor tensor_image = torch::from_blob(imagesize.data, {1, imagesize.rows * imagesize.cols},
                                                  torch::kByte).toType(torch::kFloat);
    tensor_image /= 255.;//归一化
    // 加载模型
    auto module = torch::jit::load("model/mnist.pt");
    // 声明输入
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);
    auto rst = module.forward(inputs).toTensor();
    std::cout << rst << std::endl;
    std::cout << torch::argmax(rst, 1) << std::endl;
}

