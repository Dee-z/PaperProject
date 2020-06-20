#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-20:06
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : Decoder.py
# @Project  : PaperProject
from tensorflow import keras

from Encoder import EncoderLayer
from MultiHeadAttention import MultiHeadAttention
from utils import feed_forward_network
import tensorflow as tf


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask):

        attn1, attn_weights1 = self.mha1(x, x, x, decoder_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layer_norm1(attn1 + x)

        attn2, attn_weights2 = self.mha2(out1, encoding_outputs, encoding_outputs, encoder_decoder_padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layer_norm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(ffn_output + out2)

        return out3, attn_weights1, attn_weights2


if __name__ == '__main__':
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_input = tf.random.uniform((64, 50, 512))
    sample_output = sample_encoder_layer(sample_input, False, None)
    # =============================
    sample_decoder_layer = DecoderLayer(512, 8, 2048)
    sample_decoder_input = tf.random.uniform((64, 60, 512))

    sample_decoder_output, \
    sample_decoder_attn_weights1, \
    sample_decoder_attn_weights2 = sample_decoder_layer(
        sample_decoder_input, sample_output, False, None, None)
    print('sample_decoder_output', sample_decoder_output.shape)
    print('sample_decoder_attn_weights1', sample_decoder_attn_weights1.shape)
    print('sample_decoder_attn_weights2', sample_decoder_attn_weights2.shape)
