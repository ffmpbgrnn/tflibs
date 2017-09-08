import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl

# pylint: disable=protected-access
_linear = rnn_cell_impl._linear
# pylint: enable=protected-access


def softmax_similarity(query, M, query_size, scope=None, reuse=None):
  with tf.variable_scope(scope or 'softmax_similarity', reuse=reuse) as scope:
    attn_v = tf.get_variable("AttnV_0", [query_size])
    def state_proj(h):
      return _linear([h], query_size, True)
    query = state_proj(query)
  query4d = tf.reshape(query, [-1, 1, 1, query_size])
  batch_size = tf.shape(query)[0]
  if False:
    M = tf.tile(M, tf.pack([batch_size, 1, 1, 1]))
  weight2d = tf.reduce_sum(attn_v * tf.tanh(M + query4d),
                           [2, 3])
  weight2d = tf.nn.softmax(weight2d)
  return weight2d


def cosine_similarity(query, M):
  # M <1, num_slots, 1, mem_size>
  batch_size = tf.shape(query)[0]
  M = tf.squeeze(M, [2])
  if False:
    M = tf.tile(M, tf.pack([batch_size, 1, 1]))
  query = tf.reshape(query, [batch_size, 1, -1])
  w = tf.batch_matmul(query, M, adj_y=True)
  w = tf.squeeze(w)
  w /= (tf.reduce_sum(query, [0, 1, 2]) * tf.reduce_sum(M, [0, 1, 2]) + 1e-12)
  w2d = tf.nn.softmax(w)
  return w2d
