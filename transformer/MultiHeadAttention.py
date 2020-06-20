#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-19:30
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : MultiHeadAttention.py
# @Project  : PaperProject

from tensorflow import keras
import tensorflow as tf


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = self.d_model // self.num_heads  # 64
        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.transpose(x, perm=[0, 2, 1, 3])  # large split seq_len

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)  # (batch_size, seq_len_q,depth)
        k = self.WK(k)  # (batch_size, seq_len_k, depth)
        v = self.WV(v)  # (batch_size, seq_len_v, depth)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_outputs, attention_weights =


if __name__ == '__main__':
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 256))  # (batch_size, seq_len_q, dim)
    output, attn = temp_mha(y, y, y, mask=None)
    # print(output.shape)
    # print(attn.shape)
