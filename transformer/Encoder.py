#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-19:46
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : Encoder.py
# @Project  : PaperProject
import tensorflow as tf
import keras as keras
from tensorflow import keras
from MultiHeadAttention import MultiHeadAttention
from utils import feed_forward_network


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, training, encoder_padding_maks):
        attn_output, _ = self.mha(x, x, x, encoder_padding_maks)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2


if __name__ == '__main__':
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_input = tf.random.uniform((64, 50, 512))
    sample_output = sample_encoder_layer(sample_input, False, None)
    print(sample_output.shape)
