"""Utility functions for Holparam code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from random import shuffle


class Params(dict):
  """Very simple Hyperparameter wrapper around dictionaries."""

  def __init__(self, *args, **kwargs):
    super(Params, self).__init__(*args, **kwargs)
    self.__dict__ = self


def vocab_table_from_file(filename, reverse=False):
  with tf.gfile.Open(filename, 'r') as f:
    keys = [s.strip() for s in f.readlines()]
    values = tf.range(len(keys), dtype=tf.int64)
    if not reverse:
      with tf.device('/CPU:0'):
        inits = tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
        tabl = tf.contrib.lookup.HashTable(inits, 1)
        ks = keys[:20]
        # shuffle(ks)
        tks = tf.convert_to_tensor(ks)
        tvals = tf.to_int32(tabl.lookup(tks))
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.tables_initializer())
          key_tab = tks.eval()
          val_tab = tvals.eval()
          tf.logging.info("First 20 tokens with ids")
          for (k, t) in zip(key_tab, val_tab):
            tf.logging.info("ID: %d, TOKEN: %s", t, k)
        
        return tabl
    else:
      inits = tf.contrib.lookup.KeyValueTensorInitializer(values, keys)
      return tf.contrib.lookup.HashTable(inits, '')

# x = vocab_table_from_file('vocab.txt')
