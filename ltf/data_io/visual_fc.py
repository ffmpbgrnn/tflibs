from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
from ltf.data_io.base import DataLoader
import math

FLAGS = tf.app.flags.FLAGS

class VisualFC(DataLoader):
  def __init__(self, coord):
    super(VisualFC, self).__init__(coord)
    self._fea_vec_size = FLAGS.fea_vec_size
    self._fea_pool_size = FLAGS.fea_pool_size
    self._num_segs = FLAGS.num_decoders
    self._input_len = FLAGS.max_whole_seq_len
    self._random_order = FLAGS.random_input_order
    self._do_cls = FLAGS.do_cls
    self._cls_num_classes = FLAGS.cls_num_classes

    self._skip_frame = [1, 4, 8]
    self._add_reverse = False
    self._enable_multi_scale_sample = True

  def _random_mapping(self, length):
    ranges = np.arange(0, length)
    np.random.shuffle(ranges)
    return ranges

  # This function must be called during tarining
  def _do_sample(self, data, feats, weights, batch_idx, seq_len, reversed_feats=None):
    num_frames = data.shape[0]
    if self._phase_train:
      if self._random_order:
        frame_idxs = self._random_mapping(num_frames)
        frame_idxs = frame_idxs[: seq_len]
        frame_idxs = np.sort(frame_idxs, axis=0)
      else:
        s = np.random.randint(0, num_frames - seq_len)
        frame_idxs = np.arange(s, s + seq_len)
      data = data[frame_idxs, ...]
    else:
      data = data[np.linspace(0, num_frames - 1, seq_len).astype(int), ...]
    for seq_idx in xrange(seq_len):
      feats[seq_idx][batch_idx] = data[seq_idx, ...]

    if reversed_feats is not None:
      seg_len = int(seq_len / 3)
      for seq_idx in xrange(seq_len):
        reversed_seq_idx = int(1. * seq_idx / seg_len) * seg_len + seg_len - (seq_idx % seg_len) - 1
        reversed_feats[seq_idx][batch_idx] = data[reversed_seq_idx, ...]

  def _get_all(self, data, feats, weights, batch_idx, seq_len, reversed_feats=None):
    num_frames = data.shape[0]
    if self._phase_train:
      frame_per_seg = int(1. * num_frames / self._num_segs)
      len_per_seg = int(seq_len / self._num_segs)
      acc_len, acc_frame = 0, 0
      for seg_idx in xrange(self._num_segs):
        if False:
          if seg_idx == 1:
            frame_per_seg = int(1. * num_frames / self._num_segs) * 2
            len_per_seg = int(1. * seq_len / self._num_segs) * 2
          else:
            frame_per_seg = int(1. * num_frames / self._num_segs / 2)
            len_per_seg = int(1. * seq_len / self._num_segs / 2)
        for nz_idx in xrange(frame_per_seg):
          feats[nz_idx + acc_len][batch_idx] = data[nz_idx + acc_frame, ...]
          reversed_feats[nz_idx + acc_len][batch_idx] = data[frame_per_seg - nz_idx - 1 + acc_frame, ...]
        for z_idx in xrange(frame_per_seg, len_per_seg):
          weights[z_idx + acc_len][batch_idx] = 0.
        acc_len += len_per_seg
        acc_frame += frame_per_seg
    else:
      for nz_idx in xrange(num_frames):
        feats[nz_idx][batch_idx] = data[nz_idx, ...]
        if reversed_feats is not None:
          reversed_feats[nz_idx][batch_idx] = data[num_frames - nz_idx - 1, ...]
      for z_idx in xrange(num_frames, seq_len):
        weights[z_idx][batch_idx] = 0.

  def _get_batch_num_addi(self, idxs):
    if self._phase_train:
      return len(idxs)

    num_insts = 0
    for i in idxs:
      d = self._meta_info[i]
      name = d['name']
      handle = self._name2split[name]
      data = handle[name]
      num_frames = data.shape[0]
      num_insts += int(math.ceil(1. * num_frames / self._input_len))
      if self._enable_multi_scale_sample:
        num_frames = data[::2].shape[0]
        num_insts += int(math.ceil(1. * num_frames / self._input_len))
        num_frames = data[::3].shape[0]
        num_insts += int(math.ceil(1. * num_frames / self._input_len))
    return num_insts


  def _get_data(self, idxs):
    num_insts = self._get_batch_num_addi(idxs)
    if self._add_reverse:
      num_insts *= 2
    outputs = {}
    reversed_feats, feats, weights = [], [], []
    for seq_idx in xrange(self._input_len):
      feats.append(np.zeros([num_insts, self._fea_pool_size, self._fea_vec_size], dtype=np.float32))
      reversed_feats.append(np.zeros([num_insts, self._fea_pool_size, self._fea_vec_size], dtype=np.float32))
      weights.append(np.ones([num_insts], dtype=np.float32))

    if self._do_cls:
      if FLAGS.use_sigmoid:
        labels = np.zeros([num_insts, self._cls_num_classes], dtype=np.int)
      else:
        labels = np.zeros([num_insts], dtype=np.int)
      label_weights = np.zeros([num_insts])

    vnames = []
    batch_idx = 0
    def __add_label_info(__batch_idx, __name):
      if self._do_cls and self._label_info.get(__name, False) is not False:
        labels[__batch_idx] = self._label_info[__name]
        label_weights[__batch_idx] = 1.
    for i in idxs:
      d = self._meta_info[i]
      name = d['name']

      handle = self._name2split[name]
      data = handle[name]
      num_frames = data.shape[0]
      if self._phase_train:
        __add_label_info(batch_idx, name)
        if num_frames > self._input_len:
          self._do_sample(data, feats, weights,
                          batch_idx, self._input_len, reversed_feats=reversed_feats)
        else:
          self._get_all(data, feats, weights,
                      batch_idx, self._input_len, reversed_feats=reversed_feats)
        vnames.append(name)
        batch_idx += 1
      else:
        def _append_reverse(_batch_idx, _data):
          _num_frames = _data.shape[0]
          for nz_idx in xrange(_num_frames):
            feats[nz_idx][_batch_idx] = _data[_num_frames - nz_idx - 1, ...]
          for z_idx in xrange(_num_frames, self._input_len):
            weights[z_idx][_batch_idx] = 0.

        def __multi_scale_sample(_data, _batch_idx):
          _num_frames = _data.shape[0]
          if _num_frames > self._input_len:
            for seg_id in xrange(int(math.ceil(1. * _num_frames / self._input_len))):
              data_seg = _data[seg_id * self._input_len: (seg_id + 1) * self._input_len]
              self._get_all(data_seg, feats, weights,
                            _batch_idx, self._input_len, reversed_feats=reversed_feats)
              __add_label_info(_batch_idx, name)
              vnames.append(name)
              _batch_idx += 1
              # reverse
              if self._add_reverse:
                _append_reverse(_batch_idx, data_seg)
                __add_label_info(_batch_idx, name)
                vnames.append(name)
                _batch_idx += 1
          else:
            self._get_all(_data, feats, weights,
                          _batch_idx, self._input_len, reversed_feats=reversed_feats)
            __add_label_info(_batch_idx, name)
            vnames.append(name)
            _batch_idx += 1
            if self._add_reverse:
              _append_reverse(_batch_idx, _data)
              __add_label_info(_batch_idx, name)
              vnames.append(name)
              _batch_idx += 1
          return _batch_idx

        _data = data[::1, :]
        batch_idx = __multi_scale_sample(_data, batch_idx)
        if self._enable_multi_scale_sample:
          _data = data[::2, :]
          batch_idx = __multi_scale_sample(_data, batch_idx)
          _data = data[::3, :]
          batch_idx = __multi_scale_sample(_data, batch_idx)

    outputs['inputs'] = feats
    outputs['reversed_inputs'] = reversed_feats
    outputs['weights'] = weights
    outputs['vnames'] = vnames
    if self._do_cls:
      outputs['labels'] = labels
      outputs['label_weights'] = label_weights
    return outputs
