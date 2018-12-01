#!/usr/bin/env python3
# coding: utf8

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def elmo_default:
with tf.Graph().as_default():
    # Adding module's variables to the current TensorFlow graph.
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True) # trainable=True to train the 4 scalar weights (as described in the paper)

    # The default signature, the module takes untokenized sentences as input.
    # The input tensor is a string tensor with shape [batch_size].
    # The module tokenizes each string by splitting on spaces.
    embeddings = elmo(
    ["the cat is on the mat", "dogs are in the fog"],
    signature="default",
    as_dict=False)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        print(sess.run(embeddings))

