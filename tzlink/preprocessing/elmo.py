#!/usr/bin/env python3
# coding: utf8

import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def elmo_default(context_lst_chunks):
    '''input format: list of list of this kind: ["the cat is on the mat", "dogs are in the fog"]'''
    with tf.Graph().as_default():
        logging.info('adding module\'s variables to the current TensorFlow graph...')
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False) # trainable=True to train the 4 scalar weights (as described in the paper)

        # The default signature, the module takes untokenized sentences as input.
        # The input tensor is a string tensor with shape [batch_size].
        # The module tokenizes each string by splitting on spaces.
        
        logging.info('running %s tensorflow sessions...', len(context_lst_chunks))
        for context_lst in context_lst_chunks:
            # Format of context_lst: ["the cat is on the mat", "dogs are in the fog"]
            embeddings = elmo(
            context_lst,
            signature="default",
            as_dict=False)    
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                #print(sess.run(embeddings))
                yield sess.run(embeddings)

'''
import pickle
contexts = pickle.load(open('gitig_context_lst','rb'))
mentions = pickle.load(open('gitig_mention_lst','rb'))

context_set = list(set(contexts))
context_lst_chunks = [context_set[x:x+20] for x in range(0, len(context_set), 20)]
context_elmo_emb = []

import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False) # trainable=True to train the 4 scalar weights (as described in the paper)

    # The default signature, the module takes untokenized sentences as input.
    # The input tensor is a string tensor with shape [batch_size].
    # The module tokenizes each string by splitting on spaces.
    
    count = 1
    logging.info('running %s tensorflow sessions...', len(context_lst_chunks))
    for context_lst in context_lst_chunks:
        embeddings = elmo(
        context_lst,
        signature="default",
        as_dict=False)    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            #print(sess.run(embeddings))
            context_elmo_emb.extend(sess.run(embeddings))
            log.info('machine still alive: %s',count)
            count += 1
'''
