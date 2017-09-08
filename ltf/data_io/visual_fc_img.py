from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
from ltf.data_io.base_mem import DataLoader
import math
from ltf.datasets.openimages import openimages
import random
import cPickle as pkl
import time
import h5py
import os

FLAGS = tf.app.flags.FLAGS

class VisualFC(DataLoader):
  def __init__(self, coord):
    super(VisualFC, self).__init__(coord)
    self._fea_vec_size = FLAGS.fea_vec_size
    self._fea_pool_size = FLAGS.fea_pool_size
    self._cls_num_classes = FLAGS.query_vocab_size

    self.use_sigmoid = FLAGS.use_sigmoid
    self.num_answer_candidates = FLAGS.num_answer_candidates

    self.dataset = openimages.OpenImages(FLAGS.dataset_path, phase_train=self._phase_train)
    self._label_info = self.dataset.get_label()

    self._kb_feature_dir = FLAGS.kb_feature_dir
    with open(FLAGS.kb_meta_path) as fin:
      self._kb_meta_info = pkl.load(fin)
    self._kb_meta_name_list = self._kb_meta_info['name_list']
    self._kb_meta_vocab = self._kb_meta_info['vocab']

    self._key_vec_size = self._fea_vec_size
    self._kb_name2shards = {}
    self._kb_size = len(self._kb_meta_name_list)
    self._num_slots = FLAGS.num_mem_slots

    self._kb_shuffle_idxs, self._kb_ptr = [], []
    for thread_idx in xrange(self._num_threads):
      self._kb_shuffle_idxs.append(np.arange(self._kb_size))
      self.shuffle_kb(thread_idx)
      self._kb_ptr.append(0)

    self.load_cls_data, self.load_mem_data = False, False
    if not self._phase_train or FLAGS.nn_task == 'memnn':
      self.load_cls_data = True
    if FLAGS.nn_task == 'gan' or self._phase_train:
      self.load_mem_data = True

    if self.load_mem_data:
      print('loading external memory')
      self.load_all_kb_2d()
    if self.load_cls_data:
      print('loading training data')
      self.load_all_cls_data()


  def load_all_cls_data(self):
    names = self._name2shards.keys()
    self._cls_data = {}
    for n in names:
      self._cls_data[n] = self._name2shards[n][n].value


  def load_all_kb_2d(self):
    if True:
      self._val_mem_expand2d = np.zeros([self._kb_size], dtype=np.int)
      cnt = 0
      for name in self._kb_meta_name_list:
        label_name = name.split('_')[0]
        self._val_mem_expand2d[cnt] = self._kb_meta_vocab[label_name]
        cnt += 1
      fin = h5py.File(os.path.join(self._kb_feature_dir, "merge.h5"))
      self._key_mem_expand2d = fin['feas'].value

  def load_all_kb_3d(self):
    self._kb_cat_to_ids = {}
    for name in self._kb_meta_name_list:
      label_name = name.split('_')[0]
      cat_id = self._kb_meta_vocab[label_name]
      name_list = self._kb_cat_to_ids.get(cat_id, [])
      name_list.append(name)
      self._kb_cat_to_ids[cat_id] = name_list
    bank_cnt = 0
    self._num_banks = int(self._kb_size / self._num_slots)
    self._key_mem_expand = np.zeros([self._num_banks, self._num_slots, self._key_vec_size],
                                dtype=np.float16)
    self._val_mem_expand = np.zeros([self._num_banks, self._num_slots], dtype=np.int)

    for cat_id, names in self._kb_cat_to_ids.iteritems():
      for name in names:
        self._key_mem_expand[bank_cnt % self._num_banks, int(bank_cnt / self._num_banks), :] = self._kb_name2shards[name][name]
        label_name = name.split('_')[0]
        self._val_mem_expand[bank_cnt % self._num_banks, int(bank_cnt / self._num_banks)] = self._kb_meta_vocab[label_name]
        bank_cnt += 1
    self._key_mem_expand2d = np.reshape(self._key_mem_expand, [-1, self._key_vec_size])
    self._val_mem_expand2d =  np.reshape(self._val_mem_expand, [-1])

  def shuffle_kb(self, thread_idx):
    np.random.shuffle(self._kb_shuffle_idxs[thread_idx])

  def _get_data(self, batch_idxs, thread_idx):
    if self._phase_train:
      return self._get_data_train(batch_idxs, thread_idx)
    else:
      return self._get_data_eval(batch_idxs, thread_idx)

  def _get_data_eval(self, batch_idxs, thread_idx):
    num_insts = len(batch_idxs)
    feats = np.zeros([num_insts, self._fea_vec_size], dtype=np.float32)
    labels, target_labels, target_weights = self._init_labels(num_insts)
    names = []

    for batch_offset, batch_id in enumerate(batch_idxs):
      d = self._meta_info[batch_idxs[batch_offset]]
      name = d['name']
      if False:
        feats[batch_offset, ...] = self._name2shards[name][name]
      else:
        feats[batch_offset, ...] = self._cls_data[name]
      self._set_labels(labels, target_labels, target_weights, batch_offset, name)
      names.append(name)

    outputs = {}
    outputs['inputs'] = feats
    outputs['names'] = names
    outputs['labels'] = labels
    outputs['target_weights'] = target_weights

    if self.load_mem_data:
      outputs['kb_key'] = self._key_mem_expand
      outputs['kb_val'] = self._val_mem_expand
    return outputs

  def _init_labels(self, num_insts):
    target_labels, target_weights = None, None
    if self.use_sigmoid or not self._phase_train:
      labels = np.zeros([num_insts, self._cls_num_classes], dtype=np.float32)
    else:
      target_weights = np.zeros((num_insts, self.num_answer_candidates), dtype=np.float32)
      target_labels = np.zeros((num_insts, self.num_answer_candidates), dtype=np.int32)
      labels = np.zeros((num_insts, self.num_answer_candidates), dtype=np.int32)
    return labels, target_labels, target_weights

  def _set_labels(self, labels, target_labels, target_weights, batch_offset, name):
    if self.use_sigmoid or not self._phase_train:
      labels[batch_offset] = self._label_info[name]
    elif True:
      label_info = self._label_info[name]
      num_ans = len(label_info)
      for idx in xrange(num_ans):
        labels[batch_offset][idx] = self.dataset.get_pad_id()
        target_labels[batch_offset][idx] = label_info[idx]
        target_weights[batch_offset][idx] = 1.
    elif False:
      label_info = self._label_info[name]
      num_ans = len(label_info)
      if num_ans > self.num_answer_candidates:
        num_ans = self.num_answer_candidates
      for idx in xrange(num_ans):
        labels[batch_offset][idx] = label_info[idx]
        if idx < num_ans - 1:
          target_labels[batch_offset][idx] = label_info[idx + 1]
          target_weights[batch_offset][idx] = 1.
      target_labels[batch_offset][num_ans - 1] = self.dataset.get_pad_id()
      for idx in xrange(num_ans, self.num_answer_candidates):
        labels[batch_offset][idx] = self.dataset.get_pad_id()
        target_labels[batch_offset][idx] = self.dataset.get_pad_id()
    else:
      label_info = self._label_info[name]
      pos = np.where(label_info > 0)[0]
      num_ans = len(pos)
      # TODO, remove it or not? Order matters?
      # np.random.shuffle(pos)
      if num_ans > self.num_answer_candidates:
        np.random.shuffle(pos)
        num_ans = self.num_answer_candidates
      for pidx in xrange(num_ans):
        labels[batch_offset][pidx] = pos[pidx]
        target_weights[batch_offset][pidx] = 1.

  def _get_data_train(self, batch_idxs, thread_idx):
    num_insts = len(batch_idxs)
    num_insts = FLAGS.batch_size
    outputs = {}
    if self.load_cls_data:
      feats = np.zeros([num_insts, self._fea_vec_size], dtype=np.float32)
      labels, target_labels, target_weights = self._init_labels(num_insts)
      names = []
      for batch_offset, batch_id in enumerate(batch_idxs):
        d = self._meta_info[batch_id]
        name = d['name']
        if False:
          feats[batch_offset, ...] = self._name2shards[name][name]
        else:
          feats[batch_offset, ...] = self._cls_data[name]

        if False:
          candidates = np.where(self._label_info[name] == 1)
          prob = 1. / len(candidates[0])
          for c in candidates[0]:
            labels[batch_offset][c] = prob
        else:
          self._set_labels(labels, target_labels, target_weights, batch_offset, name)
        names.append(name)

      outputs['inputs'] = feats
      outputs['names'] = names
      outputs['labels'] = labels
      outputs['target_labels'] = target_labels
      outputs['target_weights'] = target_weights

    kb_shuffle_idx = self._kb_shuffle_idxs[thread_idx]
    kb_ptr = self._kb_ptr[thread_idx]
    self._kb_ptr[thread_idx] = self._kb_ptr[thread_idx] + num_insts * self._num_slots
    if self._kb_ptr[thread_idx] > self._kb_size:
      self._kb_ptr[thread_idx] = 0
      self.shuffle_kb(thread_idx)
      kb_ptr = 0

    kb_inds = kb_shuffle_idx[kb_ptr: kb_ptr + num_insts * self._num_slots]
    if self.load_mem_data:
      kb_key = np.reshape(self._key_mem_expand2d[kb_inds, :], [num_insts, self._num_slots, self._key_vec_size])
      kb_val = np.reshape(self._val_mem_expand2d[kb_inds], [num_insts, self._num_slots])
    outputs['kb_key'] = kb_key
    outputs['kb_val'] = kb_val

    return outputs
