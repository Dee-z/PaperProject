#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-20-22:29
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : main.py
# @Project  : PaperProject
from tensorflow import keras
import tensorflow as tf

from Transformer import Transformer
from utils import create_masks, max_length, loss_function, CustomizedSchedule

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = pt_tokenizer.vocab_size + 2
target_vocab_size = en_tokenizer.vocab_size + 2

dropout_rate = 0.1

transformer = Transformer(num_layers,
                          input_vocab_size,
                          target_vocab_size,
                          max_length,
                          d_model, num_heads, dff, dropout_rate)
train_loss = keras.metrics.Mean(name='train_loss')  # 打印遍历损失
train_accuracy = keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')  # 打印便利累积的准确率
learning_rate = CustomizedSchedule(d_model)
optimizer = keras.optimizers.Adam(learning_rate,
                                  beta_1=0.9,
                                  beta_2=0.98,
                                  epsilon=1e-9)


def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True,
                                     encoder_padding_mask,
                                     decoder_mask,
                                     encoder_decoder_padding_mask)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables)
    )
    train_loss(loss)
    train_accuracy(tar_real, predictions)


def evaluate(inp_sentence):
    # 文本转ID
    input_id_sentence = [pt_tokenizer.vocab_size] \
                        + pt_tokenizer.encode(inp_sentence) + [pt_tokenizer.vocab_size + 1]
    # encoder_input.shape: (1, input_sentence_length)
    # 维度变换
    encoder_input = tf.expand_dims(input_id_sentence, 0)

    # decoder_input.shape: (1, 1)
    decoder_input = tf.expand_dims([en_tokenizer.vocab_size], 0)

    for i in range(max_length):
        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask \
            = create_masks(encoder_input, decoder_input)
        # predictions.shape: (batch_size, output_target_len, target_vocab_size)
        predictions, attention_weights = transformer(
            encoder_input,
            decoder_input,
            False,
            encoder_padding_mask,
            decoder_mask,
            encoder_decoder_padding_mask)
        # predictions.shape: (batch_size, target_vocab_size) 中间位置只取一个值, 中间维度消失
        predictions = predictions[:, -1, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1),
                               tf.int32)

        if tf.equal(predicted_id, en_tokenizer.vocab_size + 1):
            return tf.squeeze(decoder_input, axis=0), attention_weights

        decoder_input = tf.concat([decoder_input, [predicted_id]],  # predicted_id的中括号注意
                                  axis=-1)
    return tf.squeeze(decoder_input, axis=0), attention_weights  # 维度缩减


def translate(input_sentence, layer_name=''):
    result, attention_weights = evaluate(input_sentence)

    predicted_sentence = en_tokenizer.decode(
        [i for i in result if i < en_tokenizer.vocab_size])  # 对start_id, end_id 过滤

    print("Input: {}".format(input_sentence))
    print("Predicted translation: {}".format(predicted_sentence))




if __name__ == '__main__':
    translate('está muito frio aqui.')