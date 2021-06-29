import torch
# from transformers import *

# if torch.cuda.is_available():
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     device = torch.device("cuda")
#     print("current GPU :", torch.cuda.current_device())
#     print('GPU:',torch.cuda.get_device_name)
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")

import os
import glob

# from transformers import WEIGHTS_NAME
# print(WEIGHTS_NAME)
# ch = list(os.path.dirname(c) for c in sorted(glob.glob("tnews" + "/**/" + WEIGHTS_NAME, recursive=True)))
# print(ch)
from transformers import BertTokenizer, BertForTokenClassification

pretrained_weights = 'hfl/chinese-bert-wwm-ext'
# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
tokenizer = BertTokenizer.from_pretrained("./model/vocab.txt", additional_special_tokens=["tex_holder_", "1元2次方程", "单变量"])
# model = BertForTokenClassification.from_pretrained(pretrained_weights, num_labels=4)
res = tokenizer.tokenize("tex_holder_0_1元2次方程  in the equation above, what is the value of tex_holder_1_单变量")
print(res)
input_ids = torch.tensor(tokenizer.encode("tex_holder_0_1元2次方程  in the equation above, what is the value of tex_holder_1_单变量")).unsqueeze(0)
labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)
# print(input_ids.size)
# print(input_ids.size(1))
# #
# print(labels)
print(input_ids)
tmp = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
print(tmp)
# outputs = model(input_ids, labels=labels)
# loss, scores = outputs[:2]
# print(loss + '\n' + scores)
# 
# # tokenizer = BertTokenizer.from_pretrained("bert-base_chinese")
# # model = BertModel.from_pretrained("bert-base_chinese")
# # 
# # # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
# # 
# input_ids = torch.tensor([tokenizer.encode("夕小瑶的卖萌屋", add_special_tokens=True)])  
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]
#     print(last_hidden_states)
#
