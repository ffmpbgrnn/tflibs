import tensorflow as tf
from tensorflow.python.util import nest


def value_x_weight(value_states, weights):
  mem_length = value_states.get_shape()[1].value
  val_size = value_states.get_shape()[2].value
  val_hidden = tf.reshape(
      value_states, [-1, mem_length, val_size])

  def v_map(weight):
    w = tf.reshape(weight, [-1, 1, mem_length])
    v = tf.batch_matmul(w, val_hidden)
    v = tf.reshape(v, [-1, val_size])
    return v

  vs = []
  for w in weights:
    vs.append(v_map(w))
  return vs


def get_cell(self, out_proj=True):
  # cell = tf.nn.rnn_cell.BasicLSTMCell(self._num_lstm_units)
  cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self._num_lstm_units)
  if self._phase_train:
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell,
        input_keep_prob=self._dropout_keep_prob,
        output_keep_prob=self._dropout_keep_prob)
  if out_proj:
    cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self._num_lstm_units)
  return cell

def flatten(query):
  if nest.is_sequence(query):  # If the query is a tuple, flatten it.
    query_list = nest.flatten(query)
    for q in query_list:  # Check that ndims == 2 if specified.
      ndims = q.get_shape().ndims
      if ndims:
        assert ndims == 2
    query = tf.concat(1, query_list)
  return query
