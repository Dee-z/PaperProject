#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-23:26
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : tf_save_csv.py
# @Project  : PaperProject
import os

import pandas as pd
import numpy as np
import tensorflow as tf

row = 50
col = 8
path = "./data/"


def write2csv():
    data = pd.DataFrame(np.random.randint(1, 256, size=(row, col)))
    if not os.path.exists(path):
        os.mkdir(path)
    data.to_csv(path + 'tf_csv.csv', header=False, index=False)


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y


def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(  # 1对多
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,  # 1对1
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    write2csv()
