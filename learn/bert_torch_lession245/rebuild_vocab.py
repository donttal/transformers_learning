# coding:utf-8
import re
import torch
from transformers import BertTokenizer

MUSIC_DELIMITER = "∮"
underline = re.compile(r"_$")
TEX_STR = "(tex_holder_\d_)(\S+)"

# pretrained_weights = 'hfl/chinese-bert-wwm-ext'


def test_data(test_str, tokenizer_new):
    # tokenizer = BertTokenizer.from_pretrained("./model/vocab_small_new_new.txt",
    #                                           additional_special_tokens=labels_lst)

    # res = tokenizer_new.tokenize(test_str)
    # print(res)
    input_ids = torch.tensor(tokenizer_new.encode(test_str)).unsqueeze(0)
    # labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)
    # print(input_ids)
    tmp_new = tokenizer_new.convert_ids_to_tokens(input_ids.squeeze(0))
    print(tmp_new)
    return tmp_new


def new_vocab():
    # rebuilde vocab_txt
    vocab = []
    with open("./model/vocab_small_new_new.txt", 'r') as f_new:
        content = f_new.read().splitlines()
        for item in content:
            if underline.match(item):
                item = f"[underline{item}]"
            vocab.append(item)
    # print(vocab)
    with open("./model/vocab_tex.txt", 'w') as f_new_out:
        for line in vocab:
            f_new_out.write(line + "\n")


# def build_label_lst():
#     labels_lst = []
#     with open("./model/labels.txt", "r") as f:
#         for line in f:
#             # line = line.strip().replace("_", "")
#             labels_lst.append(line.strip())
#     labels_lst = sorted(labels_lst)
#     for line in labels_lst:
#         for index in range(len(labels_lst)):
#             if line != labels_lst[index] and labels_lst[index].startswith(line):
#                 labels_lst[index] = labels_lst[index][len(line):]
#     for line in labels_lst:
#         for index in range(len(labels_lst)):
#             if line != labels_lst[index] and labels_lst[index].startswith(line):
#                 labels_lst[index] = labels_lst[index][len(line):]
#     # for line in labels_lst:
#     #     for index in range(len(labels_lst)):
#     #         if line != labels_lst[index] and labels_lst[index].startswith(line):
#     #             labels_lst[index] = labels_lst[index][len(line):]
#     labels_lst = [line.replace("_", "") for line in labels_lst]
#     for i in range(len(labels_lst)):
#         if labels_lst[i].startswith("的"):
#             labels_lst[i] = labels_lst[i][1:]
#     with open("./model/tmp_new.txt", "w") as f_out:
#         for l in labels_lst:
#             f_out.write(l + '\n')
#     print(sorted(list(set(labels_lst))))
#     return sorted(list(set(labels_lst)))


def build_label_lst():
    labels_lst = []
    with open("./model/labels.txt", "r") as f:
        for line in f:
            line = "[" + line.strip().replace("_", "") + "]"
            labels_lst.append(line.strip())
    with open("./model/tmp.txt", "w") as f_out:
        for l in labels_lst:
            f_out.write(l + '\n')
    # print(sorted(list(set(labels_lst))))
    return sorted(list(set(labels_lst)))


if __name__ == '__main__':
    #  new_vocab()
    labels_lst = build_label_lst()
    with open("./model/0622_timestamp_gold_one_to_one_actions.txt", "r") as f_in, \
            open("./model/res.txt", "w") as f_out,\
        open("./model/new_labels.txt", "w") as f_out_new:
        labels_lst += ['[1元4次方程带三角函数]', '[4元1次方程带三角函数]', '[含变量行列式]', '[含变量行列式等式]', '[2元1次方程截距式带三角函数]', '[含变量公式单变量除式分子分母幂次连续]', '[复数1元2次方程]', '[复数3元1次方程]', '[2元24次方程]', '[4元4次方程]', '[3元5次方程]', '[含变量公式单变量除式分子分母幂次连续带复数]', '[1元1次方程带三角函数]', '[3元12次方程]', '[1元5次方程]', '[变量等号矩阵带三角函数]', '[1元3次方程带三角函数]', '[3元3次方程带三角函数]', '[1元4次方程]', '[2元1000次方程]', '[2元28次方程]', '[2元1次方程截距式]', '[2元2次方程带三角函数]', '[7元6次方程]', '[复数6元1次方程]', '[3元2次方程带三角函数]', '[含变量公式单变量幂次连续]', '[6元2次方程带三角函数]', '[6元3次方程]', '[1元7次方程]', '[3元4次方程]', '[变量等号矩阵]', '[1元2次方程带三角函数]', '[含变量矩阵]', '[2元3次方程带三角函数]', '[4元2次方程]', '[矩阵运算]', '[复数5元1次方程]', '[含变量公式单变量除式分子分母幂次不连续带复数]', '[5元3次方程]', '[3元6次方程]', '[1元10次方程]', '[复数9元2次方程]', '[2元6次方程]', '[2元4次方程带三角函数]', '[含变量矩阵等式]', '[无意义的符号]', '[3元3次方程]', '[复数1元1次方程]', '[复数7元1次方程]', '[2元11次方程]', '[2元21次方程]', '[7元1次方程]', '[4元3次方程]', '[含变量公式单变量除式分子分母幂次连续带三角函数]', '[含变量矩阵运算]', '[复数2元1次方程]', '[含变量行列式等式带三角函数]', '[7元2次方程]', '[复数1元1次方程带三角函数]', '[5元2次方程]', '[5元方程组5个等式]', '[2元7次方程]', '[含变量公式单变量除式分子分母幂次不连续]', '[1元20次方程]', '[3元5次方程带三角函数]', '[6元1次方程带三角函数]', '[1元8次方程]', '[科学计数法带三角函数]', '[2元15次方程]', '[2元16次方程]', '[8元1次方程]', '[复数2元4次方程]', '[5元1次方程带三角函数]', '[2元5次方程带三角函数]', '[1元3次方程]', '[2元20次方程]', '[1元11次方程]', '[行列式等式]', '[复数4元1次方程]', '[2元10次方程]', '[2元5次方程]', '[5元1次方程]', '[2元8次方程]', '[2元4次方程]', '[含变量公式单变量除式分子分母幂次不连续带三角函数]', '[7元1次方程带三角函数]', '[3元1次方程]', '[6元1次方程]', '[1元6次方程]', '[3元1次方程带三角函数]', '[2元9次方程]', '[3元2次方程]', '[2元1次方程带三角函数]', '[矩阵等式]', '[6元2次方程]', '[含变量公式单变量幂次连续带三角函数]', '[复数2元2次方程]', '[复数6元2次方程]', '[4元1次方程]', '[4元8次方程带三角函数]', '[2元3次方程]', '[含变量行列式运算]']
        for i in range(8): # tex_holder_超过8会被拒识掉
            labels_lst.append("tex_holder_"+str(i)+"_")
        # print(labels_lst)
        tokenizer = BertTokenizer.from_pretrained("./model/vocab_small_new_new.txt",
                                                  additional_special_tokens=labels_lst)
        # 测试用例
        input_ids = torch.tensor(tokenizer.encode(
            "if tex_holder_0_[函数等号公式] then tex_holder_1_[单函数] is equal to")).unsqueeze(0)
        #print(input_ids)
        tmp = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        # print(tmp)
        # rebuild
        rules_lst = []
        new_labels = []
        for line in f_in:
            line = line.strip()
            if line == "":
                if len(rules_lst) != 6:
                    print("len != 6", str(rules_lst[0]))
                    rules_lst.clear()
                    continue
                tex_holder_str = " ".join(rules_lst[-2].split())
                # print(tex_holder_str)
                result_finditer = re.finditer(TEX_STR, tex_holder_str)
                for item in result_finditer:
                    latex_content = item.group(2)
                    tmp_str = latex_content.replace("_", "")
                    print(tmp_str)
                    if "["+tmp_str+"]" not in labels_lst:
                        new_labels.append("["+tmp_str+"]")
                    tex_holder_str = tex_holder_str.replace(latex_content, tmp_str)
                    tex_holder_str = tex_holder_str.replace(tmp_str, "[" + tmp_str + "]")
                print(tex_holder_str)
                res = test_data(tex_holder_str.replace("[[", '[').replace("]]", "]"), tokenizer)
                f_out.write(MUSIC_DELIMITER.join(res) + "\n")
                rules_lst.clear()
            else:
                rules_lst.append(line)
        print(list(set(new_labels)))
