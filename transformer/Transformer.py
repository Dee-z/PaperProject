#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-22:14
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : Transformer.py
# @Project  : PaperProject
from tensorflow import keras
import tensorflow as tf

from Model import EncoderModel, DecoderModel
from utils import max_length


class Transformer(keras.Model):
    def __init__(self, num_layers, input_vocab_size, targte_vocab_size, max_length, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder_model = EncoderModel(num_layers, input_vocab_size, max_length, d_model, num_heads, dff, rate)
        self.decoder_model = DecoderModel(num_layers, targte_vocab_size, max_length, d_model, num_heads, dff, rate)
        self.final_layer = keras.layers.Dense(targte_vocab_size)

    def call(self, inp, tar, training, encoder_padding_mask,
             decoder_mask, encoder_decoder_padding_mask):
        encoder_outputs = self.encoder_model(
            inp, training, encoder_padding_mask
        )

        decoder_outputs, attention_weights = self.decoder_model(
            tar, encoder_outputs, training, decoder_mask, encoder_decoder_padding_mask
        )

        predictions = self.final_layer(decoder_outputs)
        return predictions, attention_weights


if __name__ == '__main__':
    sample_transformer = Transformer(2, 8500, 8000, max_length,
                                     512, 8, 2048, rate=0.1)
    temp_input = tf.random.uniform((64, 26))  # (batch_size, length)
    temp_target = tf.random.uniform((64, 31))

    predictions, attention_weights = sample_transformer(
        temp_input, temp_target, training=False,
        encoder_padding_mask=None,
        decoder_mask=None,
        encoder_decoder_padding_mask=None)

    print("predictions:", predictions.shape)
    for key in attention_weights:
        print(key, attention_weights[key].shape)
