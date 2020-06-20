#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-22:46
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : data_preprocess.py
# @Project  : PaperProject

import tensorflow_datasets as tfds

examples, info = tfds.load('ted_hrlr_translate/pt_to_en',
                           with_info = True,
                           as_supervised = True)

train_examples, val_examples = examples['train'], examples['validation']
print(info)

for pt, en in train_examples.take(5):
    print(pt.numpy())
    print(en.numpy())
    print()

en_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples),
    target_vocab_size = 2 ** 13)
pt_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples),
    target_vocab_size = 2 ** 13) # 8192