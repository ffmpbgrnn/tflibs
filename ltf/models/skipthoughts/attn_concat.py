import tensorflow as tf
from ltf.models.skipthoughts import utils
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


class Attention():
  def __init__(self, attn_size, query_size):
    self._v, self._attn_matrixes, self._attn_biases = [], [], []
    self._query_matrixes, self._query_biases = [], []
    self._attn_size, self._query_size = attn_size, query_size
    self._attention_vec_size = 50
    self._num_heads = 1
    with tf.variable_scope('Attention'):
      for a in xrange(self._num_heads):
        with tf.variable_scope('Attn_%d' % a):
          self._attn_matrixes.append(
              tf.get_variable("Matrix",
                              [1, 1, self._attn_size, self._attention_vec_size],
                              regularizer=slim.l2_regularizer(FLAGS.weight_decay)))
          self._attn_biases.append(tf.get_variable(
              "Bias", [self._attention_vec_size],
              initializer=tf.constant_initializer(0.0)))

        self._v.append(tf.get_variable("AttnV_%d" % a,
                                       [self._attention_vec_size]))

        with tf.variable_scope("Attention_%d" % a):
          self._query_matrixes.append(
              tf.get_variable("Matrix",
                              [self._query_size, self._attention_vec_size],
                              regularizer=slim.l2_regularizer(FLAGS.weight_decay)))
          self._query_biases.append(tf.get_variable(
              "Bias", [self._attention_vec_size],
              initializer=tf.constant_initializer(0.0)))

  def set_hidden_feat(self, attention_states):
    top_states = [tf.reshape(e, [-1, 1, self._attn_size])
                  for e in attention_states]
    self._attn_length = len(attention_states)
    attention_states = tf.concat(1, top_states)
    attention_states = tf.reshape(attention_states, [-1, self._attn_length, 1, self._attn_size])
    self._attention_states = attention_states
    self.hidden_features = []
    for a in xrange(self._num_heads):
      self.hidden_features.append(
          tf.nn.conv2d(attention_states, self._attn_matrixes[a], [1, 1, 1, 1], "SAME"))

  def get_zero_attn(self, x):
    batch_size = tf.shape(x)[0]  # Needed for reshaping.
    batch_attn_size = tf.pack([batch_size, self._attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=tf.float32)
             for _ in xrange(self._num_heads)]
    return attns

  def do_attn(self, query):
    ds = []  # Results of attention reads will be stored here.
    for a in xrange(self._num_heads):
      y = utils._linear(query, True, self._query_matrixes[a], self._query_biases[a])
      y = tf.reshape(y, [-1, 1, 1, self._attention_vec_size])
      s = tf.reduce_sum(
          self._v[a] * tf.tanh(self.hidden_features[a] + y), [2, 3])
      attn_w = tf.nn.softmax(s)
      d = tf.reduce_sum(
        tf.reshape(attn_w, [-1, self._attn_length, 1, 1]) * self._attention_states, [1, 2])
      ds.append(tf.reshape(d, [-1, self._attn_size]))
    return ds
