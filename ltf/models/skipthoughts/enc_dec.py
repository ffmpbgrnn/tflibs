from __future__ import division, print_function, absolute_import
import tensorflow as tf
from ltf.lib.rnn import rnn
from ltf.lib.rnn import rnn_cell
from ltf.lib.rnn import seq2seq
from ltf.models.skipthoughts import utils
import math
from numpy.random import choice
from tensorflow.python.util import nest
from ltf.lib.rnn import ln
import numpy as np
from ltf.lib.rnn import clockwork

slim = tf.contrib.slim


FLAGS = tf.app.flags.FLAGS


def _get_cell(self, cell_size, num_layers, scope=None, is_decoder=False):
  if False:
    cell = ln.LNGRUCell(cell_size)
  else:
    if is_decoder:
      if self._dec_use_clockwork:
        assert(False)
        cell = clockwork.ClockWorkGRUCell(cell_size, scope="Decoder")
      else:
        if self._use_lstm:
          cell = rnn_cell.BasicLSTMCell(cell_size)
        else:
          cell = rnn_cell.GRUCell(cell_size)
    else:
      if self._enc_use_clockwork:
        print('use clockwork')
        cell = clockwork.ClockWorkGRUCell(cell_size, scope=scope)
      else:
        if self._use_lstm:
          cell = rnn_cell.BasicLSTMCell(cell_size)
        else:
          cell = rnn_cell.GRUCell(cell_size)
  if self._phase_train:
    if is_decoder:
      cell = rnn_cell.DropoutWrapper(cell, 0.5, 0.5)
    else:
      cell = rnn_cell.DropoutWrapper(cell, 0.5, 0.5)
  if num_layers > 1:
    cell = rnn_cell.MultiRNNCell([cell] * num_layers)
  if not is_decoder:
    cell = rnn_cell.InputProjectionWrapper(cell, cell_size)
    cell = rnn_cell.OutputProjectionWrapper(cell, self._enc_out_size)
  return cell

def _state_slice(self, enc_state):
  enc_unit = self._enc_num_layers * self._enc_cell_size
  dec_unit = self._dec_num_layers * self._dec_cell_size

  if enc_unit != dec_unit:
    if False:
      mask = np.array([1] * dec_unit + [0] * (enc_unit - dec_unit))
      mask = tf.constant(mask, dtype=tf.bool, name="state_mask")
      mask = tf.random_shuffle(mask)
      enc_state = tf.transpose(enc_state)
      enc_state = tf.boolean_mask(enc_state, mask)
      enc_state = tf.transpose(enc_state)
      enc_state = tf.reshape(enc_state, [-1, dec_unit])
    elif True:
      enc_state = tf.slice(enc_state,
                           [0, enc_unit - dec_unit],
                           [-1, dec_unit])
    elif False:
      enc_state = slim.fully_connected(enc_state, dec_unit, normalizer_fn=None,
                                       normalizer_params=None, scope='state_transform',
                                       activation_fn=None)
  return enc_state


def go_encoder(self, enc_cell, enc_inputs, enc_sequence_length):
  outputs, enc_states = rnn.rnn(enc_cell, enc_inputs, sequence_length=enc_sequence_length,
                                    dtype=tf.float32, scope="encoder", use_step=self._enc_use_clockwork)
  enc_state = _state_slice(self, enc_states[-1])
  return outputs, enc_state, enc_states

def __dec_common(self, dec_name, enc_state, scope, reverse):
  self._dec_outs[dec_name] = []
  dec_outs = self._dec_outs[dec_name]
  if reverse:
    inputs = self._reversed_dec_inputs_dict[dec_name]
    targets = self._reversed_dec_targets_dict[dec_name]
    if not self._phase_train:
      self._attention.set_hidden_feat(self._enc_outputs_reversed)
    else:
      self._attention.set_hidden_feat(self._enc_outputs)
  else:
    inputs = self._dec_inputs_dict[dec_name]
    targets = self._dec_targets_dict[dec_name]
    self._attention.set_hidden_feat(self._enc_outputs)
  weights = self._dec_weights_dict[dec_name]
  return _decode(self, inputs, targets, enc_state, weights, None, scope, dec_outs)


def autoencoder(self, enc_state):
  dec_name = self._dec_targets_dict.keys()[1]
  with tf.variable_scope(dec_name) as scope:
    return __dec_common(self, dec_name, enc_state, scope, False)

def go_decoder(self, enc_state):
  self._dec_outs = {}
  if self._use_autoencoder:
    return autoencoder(self, enc_state)

  def __past():
    dec_name = self._dec_targets_dict.keys()[0]
    with tf.variable_scope(dec_name) as scope:
      return __dec_common(self, dec_name, enc_state, scope, True)
  def __present():
    dec_name = self._dec_targets_dict.keys()[1]
    with tf.variable_scope(dec_name) as scope:
      return __dec_common(self, dec_name, enc_state, scope, True)
  def __future():
    dec_name = self._dec_targets_dict.keys()[-1]
    with tf.variable_scope(dec_name) as scope:
      return __dec_common(self, dec_name, enc_state, scope, False)
  __empty_var = tf.constant(0.)
  def __empty():
    return __empty_var
  if self._phase_train and not self._do_cls:
    num_decoders = len(self._dec_targets_dict.keys())
    if num_decoders == 2:
      loss = tf.cond(self._bernoulli < 0.5, __past, __future)
    elif num_decoders == 3:
      loss = tf.case([(self._bernoulli < 1./num_decoders, __past),
                      (self._bernoulli < 2./num_decoders, __present),
                      (self._bernoulli >= 2./num_decoders, __future)], default=__empty)
    return loss
  else:
    loss = __past()
    loss += __future()
    loss /= 2.
    return loss

def _reconstruct_loss(logit, target):
  # Huber loss
  sigma = 2.
  delta = sigma * sigma
  d = logit - target
  if True:
    a = .5 * delta * d * d
    b = tf.abs(d) - 0.5 / delta
    l = tf.select(tf.abs(d) < (1. / delta), a, b)
  else:
    l = .5 * d * d
  # loss = tf.reduce_sum(d * d, reduction_indices=1)
  loss = tf.reduce_sum(l, reduction_indices=1)
  return loss


def _zeros(self, inp):
  zeros_dims = tf.pack([tf.shape(inp)[0], self._dec_cell_size])
  inp = tf.fill(zeros_dims, 0.0)
  inp.set_shape([None, self._dec_cell_size])
  return inp


def setup_dec_variables(self):
  with tf.variable_scope("dec_attn_in"):
    self._dec_attn_in_w = tf.get_variable("Matrix",
        [self._enc_out_size + FLAGS.fea_vec_size, self._dec_cell_size],
        regularizer=slim.l2_regularizer(FLAGS.weight_decay))
    self._dec_attn_in_b = tf.get_variable(
        "Bias", [self._dec_cell_size],
        initializer=tf.constant_initializer(0.0))
  with tf.variable_scope("dec_attn_out"):
    self._dec_attn_out_w = tf.get_variable("Matrix",
        [self._enc_out_size + self._dec_cell_size, FLAGS.fea_vec_size],
        regularizer=slim.l2_regularizer(FLAGS.weight_decay))
    self._dec_attn_out_b = tf.get_variable(
        "Bias", [FLAGS.fea_vec_size],
        initializer=tf.constant_initializer(0.0))


def _decode(self, dec_inputs, dec_targets, state, weights, signal, scope=None, dec_outs=[]):
  _dec_cell = _get_cell(self, self._dec_cell_size, self._dec_num_layers,
                        is_decoder=True)
  for i, inp in enumerate(dec_inputs):
    if i > 0:
      tf.get_variable_scope().reuse_variables()
    if i == 0:
      attns = self._attention.get_zero_attn(inp)
    if self._phase_train:
      if self._shift:
        inp = tf.zeros_like(inp) if i == 0 else dec_inputs[i - 1]

    inp = utils._linear(attns + [inp], True, self._dec_attn_in_w, self._dec_attn_in_b)
    if self._dec_use_clockwork:
      state = [i+1, state]
    dec_out, state = _dec_cell(inp, state, scope=scope)
    if self._dec_use_clockwork:
      state = state[-1]
    if self._use_lstm:
      attn_state = tf.slice(state, [0, self._dec_cell_size], [-1, self._dec_cell_size])
    else:
      attn_state = state
    attns = self._attention.do_attn(attn_state)
    dec_out = utils._linear(attns + [dec_out], True, self._dec_attn_out_w, self._dec_attn_out_b)

    dec_outs.append(dec_out)

  loss = seq2seq.sequence_loss(
      dec_outs, dec_targets, weights,
      softmax_loss_function=_reconstruct_loss)
  return loss
