#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2020-06-21-07:54
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : csy_lgy
# @File     : mnsit_test.py
# @Project  : PaperProject

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

path = './data/mnist.npz'

mnist = np.load(path)
x_train, y_train = mnist['x_train'], mnist['y_train']
