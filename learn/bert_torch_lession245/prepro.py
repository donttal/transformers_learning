# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# prepro.py：数据处理模块，负责将文本序列的句子处理成模型需要的tensor，包括不同任务的processor预处理函数。

import logging
import os
from enum import Enum
from typing import List, Optional, Union
import dataclasses
import numpy as np
import json
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from src.transformers.data.processors import DataProcessor
from src.transformers.file_utils import is_tf_available
from src.transformers.tokenization_utils import PreTrainedTokenizer

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

@dataclass
class InputExample:

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class NERInputExample:

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    tokens1: List[str]
    input_ids1: List[int]
    attention_mask1: Optional[List[int]]
    token_type_ids1: Optional[List[int]]
    label_id: Optional[Union[int, float]]
    tokens2: Optional[List[str]] = None
    input_ids2: Optional[List[int]] = None
    attention_mask2: Optional[List[int]] = None
    token_type_ids2: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class NERInputFeatures:
    tokens1: List[str]
    input_ids1: List[int]
    attention_mask1: Optional[List[int]]
    token_type_ids1: Optional[List[int]]
    label_id: Optional[List[Union[int, float]]]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class ChnSentiProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


class LCQMCProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels



class WeiboNerProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            label = line[1].split(" ")
            examples.append(NERInputExample(guid=guid, text_a=text_a,  label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[0]
            label = line[1].split(" ")
            examples.append(NERInputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


processors = {
    "lcqmc": LCQMCProcessor,
    "chnsenti": ChnSentiProcessor,
    "weiboner": WeiboNerProcessor
    }


output_modes = {
    "chnsenti": "classification",
    "lcqmc": "classification",
    "weiboner": "ner"
}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):                                                                           
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            

def convert_examples_to_features(
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int],
        label_list: List[str],
        output_mode: str
    ):

    def convert_text_to_ids(text):

        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        tokens = ["[CLS]"]+tokens[:max_length-2]+["[SEP]"]
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_length-text_len))
        attention_mask = [1]*text_len+[0]*(max_length-text_len)
        token_type_ids = [0]*max_length

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        return tokens, input_ids, attention_mask, token_type_ids

    def convert_text_to_ids_for_matching(text_a, text_b):
        tokens_a = tokenizer.tokenize(text_a)  
        tokens_b = tokenizer.tokenize(text_b)  
        if len(tokens_a) + len(tokens_b) > (max_length-3):
            _truncate_seq_pair(tokens_a, tokens_b, max_length-3)
        tokens =  ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_length-text_len))
        attention_mask = [1]*text_len+[0]*(max_length-text_len)
        token_type_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b)+1)+[0]*(max_length-text_len)
        
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        return tokens, input_ids, attention_mask, token_type_ids


    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for i in range(len(examples)): 

        if examples[i].text_b:
            tokens1, input_ids1, attention_mask1, token_type_ids1 = convert_text_to_ids_for_matching(examples[i].text_a, examples[i].text_b)
        else:
            tokens1, input_ids1, attention_mask1, token_type_ids1 = convert_text_to_ids(examples[i].text_a)
        
        if output_mode == "ner":
            label_id = [label_map["O"]]
            for j in range(len(tokens1)-2):
                label_id.append(label_map[examples[i].label[j]])
            label_id.append(label_map["O"])
            if len(label_id) < max_length:
                label_id = label_id +[label_map["O"]]*(max_length-len(label_id))
        else:
            label_id = label_map[examples[i].label]

        feature = InputFeatures(
            tokens1=tokens1,
            input_ids1=input_ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            label_id=label_id)

        features.append(feature)

        # if i<=3:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (examples[i].guid))
        #     logger.info("label_id: %s" % label_id)
        #     logger.info("tokens1: %s" % " ".join(tokens1))
        #     logger.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))
        #     logger.info("attention_mask1: %s" % " ".join([str(x) for x in attention_mask1]))
        #     logger.info("token_type_ids1: %s" % " ".join([str(x) for x in token_type_ids1]))

    return features


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank != -1 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels(args.data_dir)
    if evaluate:
        examples = (
            processor.get_test_examples(args.data_dir, args.input_test_name)
        )
    else:
        examples = (
            processor.get_train_examples(args.data_dir)
        )
    features = convert_examples_to_features(
        examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
    )

    # Convert to Tensors and build dataset
    all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
    all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1,  all_labels)

    return dataset, examples

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


def compute_metrics(preds, labels):
    return {"acc": (preds == labels).mean()}


def ner_F1(preds, labels, mask_indicators):
    assert len(preds) == len(labels) == len(mask_indicators)
    print(preds.shape)
    print(labels.shape)
    print(preds[0])
    print(labels[0])
    total_preds = []
    total_ground = []
    for i in range(len(preds)):
        num = sum(mask_indicators[i]) - 2
        total_preds.extend(preds[i][1: 1+num])
        total_ground.extend(labels[i][1: 1+num])

    refer_label = total_ground
    pred_label = total_preds
    fn = dict()
    tp = dict()
    fp = dict()
    for i in range(len(refer_label)):
        if refer_label[i] == pred_label[i]:
            if refer_label[i] not in tp:
                tp[refer_label[i]] = 0
            tp[refer_label[i]] += 1
        else:
            if pred_label[i] not in fp:
                fp[pred_label[i]] = 0
            fp[pred_label[i]] += 1
            if refer_label[i] not in fn:
                fn[refer_label[i]] = 0
            fn[refer_label[i]] += 1
    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())
    p_total = float(tp_total) / (tp_total + fp_total)
    r_total = float(tp_total) / (tp_total + fn_total)
    f_micro = 2 * p_total * r_total / (p_total + r_total)
    
    return {"f1_score": f_micro}
        


if _has_sklearn:
    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

