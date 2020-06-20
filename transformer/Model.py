#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-20:42
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : Model.py
# @Project  : PaperProject
from tensorflow import keras
import tensorflow as tf
from Encoder import EncoderLayer
from Decoder import DecoderLayer
from utils import get_position_embedding, max_length


class EncoderModel(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size, max_length, d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length
        self.embedding = keras.layers.Embedding(input_vocab_size, self.d_model)
        self.position_embedding = get_position_embedding(max_length, self.d_model)
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)
        ]

    def call(self, x, training, encoder_padding_mask):
        input_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(input_seq_len, self.max_length)

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :input_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)

        return x


class DecoderModel(keras.layers.Layer):
    def __init__(self, num_layers, target_vocab_size, max_length, d_model, num_heads, dff, rate=0.1):
        super(DecoderModel, self).__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        self.embedding = keras.layers.Embedding(target_vocab_size, self.d_model)
        self.position_embedding = get_position_embedding(max_length, self.d_model)

        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)
        ]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, encoding_outputs, training,
             decoder_mask, encoder_decoder_padding_mask):
        # x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            output_seq_len, self.max_length,
            "output_seq_len should be less or equal to self.max_length")

        attention_weights = {}

        # x.shape: (batch_size, output_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :output_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn1, attn2 = self.decoder_layers[i](
                x, encoding_outputs, training,
                decoder_mask, encoder_decoder_padding_mask)
            attention_weights[
                'decoder_layer{}_att1'.format(i + 1)] = attn1
            attention_weights[
                'decoder_layer{}_att2'.format(i + 1)] = attn2
        # x.shape: (batch_size, output_seq_len, d_model)
        return x, attention_weights


if __name__ == '__main__':
    sample_encoder_model = EncoderModel(2, 8500, max_length,
                                        512, 8, 2048)
    sample_encoder_model_input = tf.random.uniform((64, 37))
    sample_encoder_model_output = sample_encoder_model(

        sample_encoder_model_input, training=False, encoder_padding_mask=None)

    print("sample_encoder_model_output:", sample_encoder_model_output.shape)

    # ======================================================
    sample_decoder_model = DecoderModel(2, 8000, max_length,
                                        512, 8, 2048)

    sample_decoder_model_input = tf.random.uniform((64, 35))
    sample_decoder_model_output, sample_decoder_model_att \
        = sample_decoder_model(
        sample_decoder_model_input,
        sample_encoder_model_output,
        training=False, decoder_mask=None,
        encoder_decoder_padding_mask=None)

    print("sample_decoder_model_output:", sample_decoder_model_output.shape)
    for key in sample_decoder_model_att:
        print(key, ":", sample_decoder_model_att[key].shape)
