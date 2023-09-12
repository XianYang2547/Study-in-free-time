<p align="left">
  <a href [https://github.com/XianYang2547/Home-Page]">
  <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub"></a>


# 记录一下
NLP在数据和模型上比较难处理，可能是刚接触的原因<br>
[在这儿](https://huggingface.co/models)下了bert_base-chinese预训练权重，[百毒网盘](https://pan.baidu.com/s/150OiaeCRW_iJQ61G5N7clg?pwd=2547)<br>
也找了个小数据尝试了下，结果挺准的<br>
又找了个[数据集](https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip)，包含对10个类别的正负评价，我想使得网络输出正向和负向，以及评价的商品类别，通过代码修改后，在第一轮的训练中，dataloader迭代到某个地方的时候，发生了这样一个错误ValueError: Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.很疑惑，因为手头还有其他工作，测试也没去改了，先放下，回头再来看



