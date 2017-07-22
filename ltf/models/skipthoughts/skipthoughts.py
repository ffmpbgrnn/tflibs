from __future__ import division, print_function, absolute_import
import tensorflow as tf
from ltf.models.base import NNModel
from ltf.lib.rnn import rnn
from ltf.lib.rnn import rnn_cell
from ltf.lib.rnn import clockwork
from ltf.lib.rnn import ln
from ltf.lib.rnn import seq2seq

from ltf.models.skipthoughts import enc_dec
from ltf.models.skipthoughts import attn_concat as attn
from ltf.models.skipthoughts import knn
# from ltf.models.skipthoughts import ngpu
from ltf.classifier import utils as cls_utils
from ltf.models.skipthoughts import utils as model_utils
import numpy as np
import random
import h5py

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

class SkipThoughts(NNModel):
  def __init__(self,
               enc_cell_size=512, dec_cell_size=512,
               enc_seq_len=30, dec_seq_len=30,
               enc_num_layers=1, dec_num_layers=1):
    super(SkipThoughts, self).__init__()
    self._enc_seq_len, self._dec_seq_len = enc_seq_len, dec_seq_len
    assert self._enc_seq_len == self._dec_seq_len
    self._input_reverse = FLAGS.input_reverse
    self._num_decoders = FLAGS.num_decoders
    self._enc_cell_size, self._enc_num_layers = enc_cell_size, enc_num_layers
    self._dec_cell_size, self._dec_num_layers = dec_cell_size, dec_num_layers
    self._unconditional = False
    self._shift, self._use_signal_embed = True, False
    self._share_decoders = False
    self._dec_outs = []
    self._max_whole_seq_len = FLAGS.max_whole_seq_len
    self._phase_train = FLAGS.phase_train
    self._do_cls = FLAGS.do_cls
    self._cls_eval_scores = {}
    self._cls_eval_labels = {}
    self._decoder_attn = True
    self._enc_out_size = FLAGS.enc_out_size
    # self._enc_out_size = FLAGS.fea_vec_size
    self._stochastic_decoder = True
    self._use_sigmoid = FLAGS.use_sigmoid
    self._cls_num_classes = FLAGS.cls_num_classes
    self._feas_dict = None
    self.myTEST = False
    self._enc_use_clockwork = True
    self._dec_use_clockwork = False
    self._use_autoencoder = False
    self._use_lstm = False

    def ln(_input, s, b, epsilon=1e-1, max=1000):
      """ Layer normalizes a 2D tensor along its second axis, which corresponds to batch """
      m, v = tf.nn.moments(_input, [1], keep_dims=True)
      normalised_input = (_input - m) / tf.sqrt(v + epsilon)
      return normalised_input * s + b
    dim = 1024
    s1 = tf.get_variable("s1", [dim], initializer=tf.constant_initializer(1.0))
    b1 = tf.get_variable("b1", [dim], initializer=tf.constant_initializer(0.))


    self._feats_raw, self._reversed_feats_raw, self._weights = [], [], []
    self._conv_feats, self._feats, self._reversed_feats = [], [], []
    self._batch_norm_params = {'decay': 0.97, 'epsilon': 0.001, 'is_training': self._phase_train}#, 'scale': True}
    self._cl = int(1. * self._max_whole_seq_len / self._num_decoders)
    if self._phase_train and not self._do_cls:# and not self.myTEST:
      # input_len = int(self._cl * 2.5)
      input_len = 3 * self._cl
    else:
      input_len = self._max_whole_seq_len
    with tf.variable_scope("InputProj"):
      for i in xrange(input_len):
        self._weights.append(
            tf.placeholder(tf.float32, shape=[None], name="weights_{0}".format(i)))
        self._feats_raw.append(
            tf.placeholder(tf.float32, shape=[None, 1, FLAGS.fea_vec_size],
            name="feats_{0}".format(i)))
        self._reversed_feats_raw.append(
            tf.placeholder(tf.float32, shape=[None, 1, FLAGS.fea_vec_size],
            name="reversed_feats_{0}".format(i)))
        feat_temp = tf.reshape(self._feats_raw[i], [-1, FLAGS.fea_vec_size])
        reversed_feat_temp = tf.reshape(self._reversed_feats_raw[i], [-1, FLAGS.fea_vec_size])
        self._conv_feats.append(feat_temp)
        self._feats.append(feat_temp)
        self._reversed_feats.append(reversed_feat_temp)

    if self._phase_train:
      if False:
        self._enc_ind_start = 0
        self._enc_ind_end = int(self._cl * 2)
        self._dec_ind_start = [int(self._cl * 2)]
        self._dec_ind_end = [int(self._cl * 2.5)]
      else:
        self._enc_ind_start = 1 * self._cl
        self._enc_ind_end = 2 * self._cl
        self._dec_ind_start = [0 * self._cl, 2 * self._cl]
        self._dec_ind_end = [1 * self._cl, 3 * self._cl]
    else:
      self._enc_ind_start = 0
      self._enc_ind_end = self._max_whole_seq_len
      self._dec_ind_start = [0 * self._cl, 0 * self._cl]
      self._dec_ind_end = [3 * self._cl, 3 * self._cl]

    self._enc_inputs = self._feats[self._enc_ind_start: self._enc_ind_end]
    self._enc_weights = self._weights[self._enc_ind_start: self._enc_ind_end]

    self._dec_inputs_dict, self._dec_targets_dict, self._dec_weights_dict = {}, {}, {}
    self._reversed_dec_inputs_dict, self._reversed_dec_targets_dict = {}, {}
    for i in xrange(len(self._dec_ind_start)):
      dec_name = 'decoder_%d' % i
      s, e = self._dec_ind_start[i], self._dec_ind_end[i]
      self._dec_inputs_dict[dec_name] = self._feats[s: e]
      self._dec_targets_dict[dec_name] = self._conv_feats[s: e]
      self._reversed_dec_inputs_dict[dec_name] = self._reversed_feats[s: e]
      self._reversed_dec_targets_dict[dec_name] = self._reversed_feats[s: e]
      self._dec_weights_dict[dec_name] = self._weights[s: e]

    for i in xrange(self._enc_ind_start, self._enc_ind_end):
      if i == self._enc_ind_start:
        self._enc_sequence_length = self._weights[i]
      else:
        self._enc_sequence_length += self._weights[i]
    self._enc_sequence_length = tf.to_int32(self._enc_sequence_length)

    if self._do_cls:
      if self._use_sigmoid:
        self._labels = tf.placeholder(tf.float32, shape=[None, self._cls_num_classes], name="cls_labels")
      else:
        self._labels = tf.placeholder(tf.int32, shape=[None], name="cls_labels")
      self._label_weights = tf.placeholder(tf.float32, shape=[None], name="cls_label_weights")

    with tf.variable_scope('encoder_disc'):
      enc_cell = enc_dec._get_cell(self, self._enc_cell_size, self._enc_num_layers, scope="Encoder_Layer_0")

      self._bernoulli = tf.random_uniform([1], 0, 1)[0]
      def __reverse():
        return self._reversed_feats[self._enc_ind_start: self._enc_ind_end] # self._enc_inputs[::-1]
      def __empty():
        return self._enc_inputs

      if self._phase_train and not self._do_cls:
        if self._use_autoencoder:
          self._enc_outputs, self._enc_state, _ = enc_dec.go_encoder(
              self, enc_cell, self._enc_inputs, self._enc_sequence_length)
        else:
          num_decoders = len(self._dec_targets_dict)
          self._enc_inputs = tf.cond(self._bernoulli < 1./num_decoders, __reverse, __empty)
          self._enc_outputs, self._enc_state, _ = enc_dec.go_encoder(
              self, enc_cell, self._enc_inputs, self._enc_sequence_length)
          self._enc_state = tf.nn.dropout(self._enc_state, 0.5)
      else:
        self._enc_outputs, self._enc_state, self._enc_states = enc_dec.go_encoder(
            self, enc_cell, self._enc_inputs, self._enc_sequence_length)
        tf.get_variable_scope().reuse_variables()
        self._enc_outputs_reversed, self._enc_state_reversed, _ = enc_dec.go_encoder(
            self, enc_cell, __reverse(), self._enc_sequence_length)

    self._attention = attn.Attention(self._enc_out_size, self._dec_cell_size * self._dec_num_layers)
    enc_dec.setup_dec_variables(self)

    if self._do_cls:
      self._vid_rep = self._enc_state # self._enc_outputs[-1]# + self._enc_inputs[-1]
      loss = 0
      if self._phase_train:
        if True:
          self._reconstruct_loss = tf.constant(0.)
        else:
          self._reconstruct_loss = enc_dec.go_decoder(self, self._enc_state)
        self._restore_vars = tf.all_variables()
        self._cls_logit, self._cls_loss = self._cls_task(self._vid_rep, self._labels, self._label_weights)
        loss = self._cls_loss + self._reconstruct_loss * 0.01
      else:
        self._enc_outputs = self._enc_states
    else:
      self._reconstruct_loss = enc_dec.go_decoder(self, self._enc_state)
      self._cls_loss = tf.constant(0.)
      loss = self._reconstruct_loss
    self._total_loss = loss
    if self._phase_train:
      self._get_train_op()


  def _do_cls_mlp(self, vid_rep, early_return=False):
    with tf.variable_scope('cls_fc0'):
      cls_logits = rnn_cell._linear([vid_rep], FLAGS.fea_vec_size, True,
                                          weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay))
      cls_logits = tf.nn.relu(cls_logits)
      if self._phase_train:
        cls_logits = tf.nn.dropout(cls_logits, 0.5)
    with tf.variable_scope('cls_fc1'):
      cls_logits = rnn_cell._linear([cls_logits], FLAGS.fea_vec_size, True,
                                          weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay))
      cls_logits = tf.nn.relu(cls_logits)
      if self._phase_train:
        cls_logits = tf.nn.dropout(cls_logits, 0.5)
    with tf.variable_scope('cls_fc2'):
      cls_logits = rnn_cell._linear([cls_logits], self._cls_num_classes, True,
                                          weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay))
    return cls_logits

  def _cls_task(self, vid_rep, labels=None, label_weights=None):
    '''label_weights [batch_size]'''
    cls_logits = self._do_cls_mlp(vid_rep)

    cross_entropy_mean = None
    if self._phase_train:
      if self._use_sigmoid:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            cls_logits, labels, name='cross_entropy_per_example')
        label_weights = tf.reshape(label_weights, [-1, 1])
      else:
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            cls_logits, labels, name='cross_entropy_per_example')
      cross_entropy = tf.mul(cross_entropy, label_weights)
      cross_entropy_mean = tf.reduce_sum(cross_entropy) / tf.reduce_sum(label_weights)
    else:
      if self._use_sigmoid:
        cls_logits = tf.nn.sigmoid(cls_logits)
      else:
        cls_logits = tf.nn.softmax(cls_logits)
    return cls_logits, cross_entropy_mean

  def _get_input_feed_dict(self, batch):
    input_feed = {}
    enc_sequence_length = None
    if True:
      for l in xrange(self._max_whole_seq_len):
        input_feed[self._feats_raw[l].name] = batch['inputs'][l]
        input_feed[self._weights[l].name] = batch['weights'][l]
        input_feed[self._reversed_feats_raw[l].name] = batch['reversed_inputs'][l]
    else:
      # send to encoder
      enc_len = self._cl# * 2
      for l in xrange(enc_len):
        dl = l + int(self._cl)# / 2)
        input_feed[self._feats_raw[l].name] = batch['inputs'][dl]
        input_feed[self._weights[l].name] = batch['weights'][dl]
        if enc_sequence_length is None:
          enc_sequence_length = batch['weights'][dl]
        else:
          enc_sequence_length += batch['weights'][dl]
      enc_sequence_length = enc_sequence_length.astype(np.int)
      input_feed[self._enc_sequence_length.name] = enc_sequence_length

      # send to decoder, index 1
      bl = random.randint(0, 1)
      dec_start = 0 if bl == 0 else int(self._cl * 2)#.5)
      dec_len = int(self._cl)# / 2)
      for l in xrange(dec_len):
        dl = l + dec_start
        l = l + enc_len
        input_feed[self._feats_raw[l].name] = batch['inputs'][dl]
        input_feed[self._weights[l].name] = batch['weights'][dl]
    if self._do_cls:
      input_feed[self._labels.name] = batch['labels']
      input_feed[self._label_weights.name] = batch['label_weights']
    return input_feed

  def train_step(self, sess, batch, with_train_op=True):
    input_feed = self._get_input_feed_dict(batch)
    if with_train_op:
      op = self._train_op
      # check_op = tf.add_check_numerics_ops()
      output_feed = [op, self._total_loss, self._cls_loss, self._reconstruct_loss, self._global_norm]
      loss = sess.run(output_feed, input_feed)
      return loss[1:]
    else:
      op = self._summary_op
      output_feed = [op, self._total_loss]
      _str, _ = sess.run(output_feed, input_feed)
      return _str

  def save_feat(self, states, batches, save_dict):
    for step_idx, one_step in enumerate(states):
      for batch_idx, name in enumerate(batches['vnames']):
        if int(batches['weights'][step_idx][batch_idx]) == 1:
          t = save_dict.get(name, [])
          if len(t) == 0:
            t.append(np.reshape(one_step[batch_idx], [-1]))
            t.append(1)
          else:
            t[0] = t[0] + np.reshape(one_step[batch_idx], [-1])
            t[1] = t[1] + 1
          save_dict[name] = t


  def write_feat(self):
    if len(FLAGS.feature_extraction_dir) > 0:
      for idx in self._feas_dict.keys():
        if idx == 0:
          fout = h5py.File(FLAGS.feat_save_path, 'w')
        else:
          fout = h5py.File(FLAGS.feat_save_path+'_OUT{0}'.format(idx), 'w')
        save_dict = self._feas_dict[idx]
        for vname, d in save_dict.iteritems():
          d = d[0] / d[1]
          fout.create_dataset(vname, data=d)
        fout.close()


  def eval_step(self, sess, batch, fout=None):
    input_feed = self._get_input_feed_dict(batch)
    if self._do_cls and False:
      output_feed = self._cls_logits
      states = sess.run(output_feed, input_feed)
    else:
      if False:
        output_feed = [self._enc_outputs, self._enc_outputs_reversed]
        for dec_name, dec_out in self._dec_outs.iteritems():
          output_feed.append(dec_out)
      else:
        output_feed = [self._enc_outputs]

      states = sess.run(output_feed, input_feed)
      if self._feas_dict is None:
        self._feas_dict = {}
        for idx in xrange(len(states)):
          self._feas_dict[idx] = {}
      for idx, state in enumerate(states):
        self.save_feat(state, batch, self._feas_dict[idx])
      return
    if self._do_cls:
      scores = states
      labels, label_weights = batch['labels'], batch['label_weights']
      for idx in xrange(label_weights.shape[0]):
        weight = label_weights[idx]
        if weight > 0:
          score = scores[idx, :][None, :]
          vid = batch['vnames'][idx]
          s = self._cls_eval_scores.get(vid, [])
          if len(s) == 0:
            s.append(score)
            s.append(1)
          else:
            s[0] = s[0] + score
            s[1] = s[1] + 1
          self._cls_eval_scores[vid] = s
          self._cls_eval_labels[vid] = labels[idx]

  def eval_get_metric(self, fout):
    if not self._do_cls:
      return
    return
    scores, labels = [], []
    for vid in self._cls_eval_labels:
      s = self._cls_eval_scores[vid]
      s = s[0] / s[1]
      scores.append(s)
      labels.append(self._cls_eval_labels[vid])
    scores = np.vstack(scores)
    labels = np.array(labels)
    _map = 0.
    if self._use_sigmoid:
      label_offset = 0
    else:
      label_offset = 1
    for label_idx in xrange(label_offset, int(scores.shape[1])):
      ap = cls_utils.get_ap(scores[:, label_idx], labels, label_id=label_idx, one_hot_label=False)
      _map += ap
    _str = "mAP: {0}".format(_map / (scores.shape[1] - 1))
    fout.write(_str)
    fout.flush()
