# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 14:46
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : use_Tokenizer.py
# ------❤️❤️❤️------ #

from transformers import BertTokenizer

# 加载分词工具
Tokenizer = BertTokenizer.from_pretrained("model/vocab.txt")
# print(token)

sents = ['位置尚可，但距离海边的位置比预期的要差的多',
         '5月8日付款成功，当当网显示5月10日发货，可是至今还没看到货物，也没收到任何通知，简不知怎么说好！！！',
         '整体来说，本书还是不错的。至少在书中描述了许多现实中存在的司法系统方面的问题，这是值得每个法律工作者去思考的。尤其是让那些涉世不深的想加入到律师队伍中的年青人，看到了社会特别是中国司法界真实的一面。缺点是：书中引用了大量的法律条文和司法解释，对于已经是律师或有一定工作经验的法律工作者来说有点多余，而且所占的篇幅不少，有凑字数的嫌疑。整体来说还是不错的。不要对一本书提太高的要求。']

# 批量编码句子
out = Tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,
    # 当句子长度大于max_length时，截断
    truncation=True,
    max_length=30,
    # 一律补0到max_length长度
    padding="max_length",
    # 可取值：tf,pt,np,默认为返回list
    return_tensors=None,
    # 返回attention_mask
    return_attention_mask=True,
    # 返回token_type_ids
    return_token_type_ids=True,
    # 返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,
    # 返回offsets_mapping 标识每个词的起止位置，这个参数只能在BertTokenizerFast使用
    # return_offsets_mapping=True,
    # 返回length标识长度
    return_length=True,
)
# print(out)
for k, v in out.items():
    print(k, ":", v)

print(Tokenizer.decode(out["input_ids"][0]), Tokenizer.decode(out["input_ids"][1]))

# 获取字典
vocab = Tokenizer.get_vocab()
print(type(vocab), len(vocab), "阳光" in vocab)
# 添加新词
Tokenizer.add_tokens(new_tokens=["阳光", "大地"])
# 添加新符号
Tokenizer.add_special_tokens({"eos_token": "[EOS]"})
vocab = Tokenizer.get_vocab()
print(type(vocab), len(vocab), "阳光" in vocab, "[EOS]" in vocab)
# 编码新句子
out = Tokenizer.encode(
    text="阳光洒在大地上[EOS]",
    text_pair=None,
    truncation=True,
    padding="max_length",
    max_length=10,
    add_special_tokens=True,
    return_tensors=None
)
print(out)
print(Tokenizer.decode(out))
