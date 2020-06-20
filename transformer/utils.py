#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-19:30
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : utils.py
# @Project  : PaperProject
import tensorflow as tf
import numpy as np
from tensorflow import keras

max_length = 40


def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


def scaled_dot_product_attention(q, k, v, mask):
    """
    Args:
    - q: shape == (..., seq_len_q, depth)
    - k: shape == (..., seq_len_k, depth)
    - v: shape == (..., seq_len_v, depth_v)
    - seq_len_k == seq_len_v
    - mask: shape == (..., seq_len_q, seq_len_k)
    Returns:
    - output: weighted sum
    - attention_weights: weights of attention
    """

    # matmul_qk.shape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        # 使得在softmax后值趋近于0
        scaled_attention_logits += (mask * -1e9)

    # attention_weights.shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)

    # output.shape: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def print_scaled_dot_product_attention(q, k, v):
    temp_out, temp_att = scaled_dot_product_attention(q, k, v, None)
    print("Attention weights are:")
    print(temp_att)
    print("Output is:")
    print(temp_out)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def feed_forward_network(d_model, dff):
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000,
                               (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def get_position_embedding(sentence_length, d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # sines.shape: [sentence_length, d_model / 2]
    # cosines.shape: [sentence_length, d_model / 2]
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    # position_embedding.shape: [sentence_length, d_model]
    position_embedding = np.concatenate([sines, cosines], axis=-1)
    # position_embedding.shape: [1, sentence_length, d_model]
    position_embedding = position_embedding[np.newaxis, ...]

    return tf.cast(position_embedding, dtype=tf.float32)


if __name__ == '__main__':
    # x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    # create_padding_mask(x)
    #
    # create_look_ahead_mask(3)
    #
    # temp_k = tf.constant([[10, 0, 0],
    #                       [0, 10, 0],
    #                       [0, 0, 10],
    #                       [0, 0, 10]], dtype=tf.float32)  # (4, 3)
    #
    # temp_v = tf.constant([[1, 0],
    #                       [10, 0],
    #                       [100, 5],
    #                       [1000, 6]], dtype=tf.float32)  # (4, 2)
    #
    # temp_q1 = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    # np.set_printoptions(suppress=True)
    # print_scaled_dot_product_attention(temp_q1, temp_k, temp_v)
    #
    # temp_q2 = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    # print_scaled_dot_product_attention(temp_q2, temp_k, temp_v)
    #
    # temp_q3 = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
    # print_scaled_dot_product_attention(temp_q3, temp_k, temp_v)
    #
    # temp_q4 = tf.constant([[0, 10, 0],
    #                        [0, 0, 10],
    #                        [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    # print_scaled_dot_product_attention(temp_q4, temp_k, temp_v)

    sample_ffn = feed_forward_network(512, 2048)
    sample_ffn(tf.random.uniform((64, 50, 512)))
