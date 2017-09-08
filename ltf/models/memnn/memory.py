import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops

slim = tf.contrib.slim

'''
if self.two_bank:
  self.val_size = descs_states.get_shape()[2].value
  self.val_hidden = array_ops.reshape(
      descs_states, [-1, self.mem_length, 1, self.val_size])
  v = math_ops.reduce_sum(
      array_ops.reshape(w, [-1, self.mem_length, 1, 1]) * self.val_hidden,
      [1, 2])
  vs.append(array_ops.reshape(v, [-1, self.val_size]))
'''


class Memory(object):
  def __init__(self, key_size, query_size, scope, enable_query=True):
    # mem_state: <batch_size, num_slots, mem_size>
    self.num_heads = 1
    self.attention_vec_size = key_size
    self.key_size = key_size
    self.enable_query = enable_query

    self.v = []
    self.memory_w = []
    self.query_w, self.query_b = [], []
    with variable_scope.variable_scope(scope+"_Memory_Module"):
      for a in xrange(self.num_heads):
        self.memory_w.append(
            tf.get_variable("AttnW_%d" % a,
                            [1, 1, self.key_size, self.attention_vec_size]))
        self.v.append(
            tf.get_variable("AttnV_%d" % a, [self.attention_vec_size]))

        self.query_w.append(
            tf.get_variable("Query_W_%d" % a, [query_size, self.attention_vec_size]))
        self.query_b.append(tf.get_variable(
            "Query_B", [self.attention_vec_size], dtype=tf.float32,
            initializer=init_ops.constant_initializer(
                0., dtype=tf.float32)))


  def get_memory(self):
    return self.memory


  def set_memory(self, mem_states):
    self.memory = mem_states
    self.mem_length = mem_states.get_shape()[1].value
    if self.mem_length is None:
      self.mem_length = tf.shape(mem_states)[1]

    if self.enable_query:
      # self.key_size = mem_states.get_shape()[2].value
      self.key_hidden = array_ops.reshape(
          mem_states, [-1, self.mem_length, 1, self.key_size])

      self.hidden_features = []
      for a in xrange(self.num_heads):
        self.hidden_features.append(
            nn_ops.conv2d(self.key_hidden, self.memory_w[a], [1, 1, 1, 1], "SAME"))

  def zero_memory(self, batch_size):
    ks = []
    for a in xrange(self.num_heads):
      ks.append(tf.zeros([batch_size, self.key_size],
                         dtype=tf.float32))
    return ks

  def zero_input(self, batch_size):
    inputs = tf.zeros(tf.pack([batch_size, self.key_size]),
                      dtype=tf.float32)
    return inputs

  def get_num_slots(self):
    return self.mem_length

  def get_mem_size(self):
    return self.key_size

  def write(self, weights, d):
    if False:
      d = tf.reshape(d, [-1, 1, self.get_mem_size()])
      weighted_d = tf.reshape(weights, [-1, self.get_num_slots(), 1]) * d
      weighted_d = tf.reshape(tf.reduce_mean(weighted_d, 0), [1, self.get_num_slots(), self.get_mem_size()])
      weights = tf.reshape(tf.reduce_mean(weights, 0), [1, -1, 1])
      memory = self.memory * (1 - weights) + weighted_d
      self.set_memory(memory)
    else:
      erase_vec = slim.fully_connected(d, self.key_size,
                                       activation_fn=tf.nn.sigmoid,
                                       scope="write_erase_head")
      add_vec = slim.fully_connected(d, self.key_size,
                                     activation_fn=None,
                                     scope="write_add_head")
      erase_vec = tf.reshape(tf.reduce_mean(erase_vec, 0), [1, 1, -1])
      add_vec = tf.reshape(tf.reduce_mean(add_vec, 0), [1, 1, -1])
      weights = tf.reshape(tf.reduce_mean(weights, 0), [1, -1, 1])
      memory = self.memory * (1 - weights * erase_vec) + weights * add_vec
      self.set_memory(memory)

  def query(self, query):
    ks, ws = [], []  # Results of attention reads will be stored here.
    if nest.is_sequence(query):  # If the query is a tuple, flatten it.
      query_list = nest.flatten(query)
      for q in query_list:  # Check that ndims == 2 if specified.
        ndims = q.get_shape().ndims
        if ndims:
          assert ndims == 2
      query = array_ops.concat(1, query_list)

    sigmoid_gates = []
    for a in xrange(self.num_heads):
      y = tf.matmul(query, self.query_w[a]) + self.query_b[a]
      y = array_ops.reshape(y, [-1, 1, 1, self.attention_vec_size])
      # Attention mask is a softmax of v^T * tanh(...).
      s = math_ops.reduce_sum(
          self.v[a] * math_ops.tanh(self.hidden_features[a] + y), [2, 3])
      w = nn_ops.softmax(s)
      sigmoid_gates.append(tf.sigmoid(s))
      ws.append(w)
      # Now calculate the attention-weighted vector d.
      k = math_ops.reduce_sum(
          array_ops.reshape(w, [-1, self.mem_length, 1, 1]) * self.key_hidden,
          [1, 2])
      ks.append(array_ops.reshape(k, [-1, self.key_size]))

    return ks, ws, ws
