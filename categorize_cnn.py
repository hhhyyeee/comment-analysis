import numpy as np
import pandas as pd
import csv
import glowpick_comment_tokenizer as gt
import urllib.request
import matplotlib.pyplot as plt
from collections import OrderedDict
import tensorflow as tf

sydney = list()
with open('csv/sydney_comments_1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        sydney.append(row)

keywords = list()
with open('csv/tokenize_keywords.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        keywords.append(row[0])

sydney_comments = [row[1] for row in sydney]
tokenizer = gt.GlowpickCommentTokenizer(sydney_comments, keywords)
sydney_tokenized = tokenizer.tokenize()

cnn_sydney = []
for comment in sydney_tokenized:
    cnn_sydney.append(comment.split(' '))

vocab = OrderedDict()
i = 0
for comment in cnn_sydney:
    for token in comment:
        if token not in vocab:
            vocab[token] = i
            i += 1

print("vocab done")

comvec_list = []
for comment in cnn_sydney:
    comvec = []
    for token in comment:
        comvec.append(vocab[token])
    comvec_list.append(comvec)

print("vectorisation done")

vocab_size = len(vocab)
embedding_size = 50

W = tf.Variable(tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0), name="W") # minval, maxval
embedded_chars = tf.nn.embedding_lookup(W, comvec_list[0]) # 차원수: [sequence length, embedding size]
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) # 차원수 : [sequence length * embedding size * 1]

num_filters = 1

conv = tf.nn.conv2d(embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")