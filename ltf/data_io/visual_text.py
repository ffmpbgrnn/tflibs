from __future__ import division, print_function, absolute_import
from ltf.data_io.base_mem import DataLoader
import cPickle as pkl
import numpy as np
import os
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
class VisualText(DataLoader):
  def _set_num_instances(self):
    self._text_data = self._dataset[FLAGS.stage]
    self.vids = []
    for d in self._text_data:
      self.vids.append(d['vname'])
    self.vids = list(set(self.vids))
    self._meta_info = self.vids
    if self._phase_train:
      self._num_instances = len(self._text_data)
    else:
      self._num_instances = len(self.vids)

  def __init__(self, coord):
    super(VisualText, self).__init__(coord)
    self._dec_seq_len = FLAGS.dec_seq_len
    self._enc_seq_len = FLAGS.enc_seq_len
    self._fea_vec_size = FLAGS.fea_vec_size
    self.vocab = self._dataset['vocab']

    self.PAD_ID = FLAGS.PAD_ID
    self.GO_ID = FLAGS.GO_ID
    self.EOS_ID = FLAGS.EOS_ID
    self.UNK_ID = FLAGS.UNK_ID

  def get_vid_feat(self, vnames, return_array=False):
    feats = []
    bs = len(vnames)
    for seq_idx in xrange(self._enc_seq_len):
      feats.append(np.zeros([bs, 1, self._fea_vec_size]))
    for batch_idx, vname in enumerate(vnames):
      handle = self._name2shards[vname]
      data = handle[vname]
      num_frames = data.shape[0]
      if num_frames > self._enc_seq_len:
        if self._phase_train:
          frame_idxs = np.arange(0, num_frames)
          np.random.shuffle(frame_idxs)
          frame_idxs = frame_idxs[: self._enc_seq_len]
          frame_idxs = np.sort(frame_idxs, axis=0)
          for seq_num, seq_idx in enumerate(frame_idxs):
            feats[seq_num][batch_idx] = data[seq_idx, ...]
        else:
          fea = data[np.linspace(0, num_frames - 1, self._enc_seq_len).astype(int), ...]
          for seq_idx in xrange(self._enc_seq_len):
            feats[seq_idx][batch_idx] = fea[seq_idx, ...]
      else:
        for seq_idx in xrange(num_frames):
          feats[seq_idx][batch_idx] = data[seq_idx, ...]
    if return_array:
      feats = np.concatenate(feats, axis=0)
    return feats

  def _get_data(self, batch_idxs, thread_idx):
    if self._phase_train:
      return self._get_data_train(batch_idxs, thread_idx)
    else:
      return self._get_data_eval(batch_idxs, thread_idx)

  def _get_data_train(self, idxs, thread_idx):
    vnames = []
    captions = []
    bs = len(idxs)
    for batch_idx in idxs:
      instance = self._text_data[batch_idx]
      vnames.append(instance['vname'])
      caption = instance['caption_inputs']
      if len(caption) > self._dec_seq_len - 2:
        caption = [self.GO_ID] + caption[: self._dec_seq_len - 2] + [self.EOS_ID]
      else:
        caption = [self.GO_ID] + caption + [self.EOS_ID]
      dec_pad_size = self._dec_seq_len - len(caption)
      captions.append(caption + [self.PAD_ID] * dec_pad_size)

    # batch x seq x pool x fea_vec
    feats = self.get_vid_feat(vnames)
    dec_inputs = []
    target_weight = []
    for seq_id in xrange(self._dec_seq_len):
      dec_input = []
      weight = np.ones(bs, dtype=np.float32)
      for batch_id in xrange(bs):
        dec_input.append(captions[batch_id][seq_id])

        if seq_id < self._dec_seq_len - 1:
          target = captions[batch_id][seq_id + 1]
        if seq_id == self._dec_seq_len - 1 or target == self.PAD_ID:
          weight[batch_id] = 0.0
      target_weight.append(weight)
      dec_inputs.append(dec_input)
    outputs = {}
    outputs['inputs'] = feats
    outputs['names'] = vnames
    outputs['decoder_inputs'] = dec_inputs
    outputs['target_weights'] = target_weight
    return outputs

  def _get_data_eval(self, idxs, thread_idx):
    assert len(idxs) == 1
    instance = self.vids[idxs[0]]
    vnames = [instance]
    feats = self.get_vid_feat(vnames)

    outputs = {}
    outputs['inputs'] = feats
    outputs['names'] = vnames
    outputs['vocab'] = self.vocab
    return outputs
