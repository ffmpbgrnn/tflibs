'''
Part code is modified from tflearn.
'''
from __future__ import division, print_function, absolute_import
import h5py
import sys
import re
import cPickle as pkl
import math
import numpy as np
import Queue as queue
import threading
import tensorflow as tf
import os
from ltf.datasets.MED import med
import cPickle as pkl
import random

FLAGS = tf.app.flags.FLAGS

class DataLoader(object):
  def __init__(self, coord):
    self._coord = coord
    with open(FLAGS.dataset_path) as fin:
      datasets = pkl.load(fin)

    meta_info = datasets[FLAGS.split_name]
    self._meta_info = meta_info[FLAGS.stage]
    self._phase_train = FLAGS.phase_train
    self._num_instances = len(self._meta_info)
    self._batch_size = FLAGS.batch_size
    self._num_threads = FLAGS.num_loaders
    self._buckets = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40)]

    self._do_cls = FLAGS.do_cls
    self._pos_neg_sample = False
    if self._phase_train and self._do_cls:
      self._pos_neg_sample = True

    self._nb_batch = int(np.ceil(self._num_instances/ float(self._batch_size)))
    if self._pos_neg_sample:
      self._num_instances = self._batch_size * self._nb_batch
    else:
      self._shuffle_idx = np.arange(self._num_instances)

    self._batches = [(i*self._batch_size, min(self._num_instances, (i+1) * self._batch_size))
            for i in range(0, self._nb_batch)]

    if self._do_cls:
      med_dataset = med.MED(FLAGS.label_info_path, 'train' if self._phase_train else 'test')
      self._label_info = med_dataset.get_label(FLAGS.use_sigmoid)
    if self._pos_neg_sample:
      self._pos_neg_info = med_dataset.cats_to_names()
      self._all_vnames = []
      for i in xrange(len(self._meta_info)):
        self._all_vnames.append(self._meta_info[i]['name'])
      self._vid_neg = self._pos_neg_info[0]
      self._vid_pos = []
      num_classes = len(self._pos_neg_info)
      for i in xrange(1, num_classes):
        self._vid_pos += self._pos_neg_info[i]
      max_pos_neg_rate = 2
      pos_neg_rate = int(len(self._vid_neg) / len(self._vid_pos))
      if pos_neg_rate > max_pos_neg_rate:
        pos_neg_rate = max_pos_neg_rate
      print(pos_neg_rate)
      self._pos_num = int(self._batch_size / (pos_neg_rate + 1))
      self._neg_num = self._batch_size - self._pos_num

    # Queue holding batch ids
    self._batch_ids_queue = queue.Queue(FLAGS.max_queue_size * 5)
    self._feed_queue = queue.Queue(FLAGS.max_queue_size)

    self._feas_in_handles = []
    self._name2split = {}
    feas_in_names = []
    for n in os.listdir(FLAGS.visual_feature_dir):
      feature_path = os.path.join(FLAGS.visual_feature_dir, n)
      h = h5py.File(feature_path)
      self._feas_in_handles.append(h)
      d = {}
      for k in h.keys():
        d[k] = True
      feas_in_names.append(d)
    for instance in self._meta_info:
      name = instance['name']
      found = False
      for p_idx, names in enumerate(feas_in_names):
        if names.get(name, False):
          self._name2split[name] = self._feas_in_handles[p_idx]
          found = True
          break
      if not found:
        print("key %s not found." % name)
        sys.stdout.flush()
        os._exit(0)

    self._data_status = DataStatus(self._batch_size, self._num_instances)
    self._data_status.reset()

    self._reset_batches()

  def start(self):
    bi_threads = [threading.Thread(target=self._fill_batch_ids_queue)]
    fd_threads = [threading.Thread(target=self._fill_feed_queue)
                  for i in range(self._num_threads)]
    self._threads = bi_threads + fd_threads
    for t in self._threads:
      t.start()

  def _fill_batch_ids_queue(self):
    while not self._coord.should_stop():
      ids = self._next_batch_ids()
      if ids is False:
        for thread_idx in xrange(self._num_threads):
          self._batch_ids_queue.put(False)
        break
      self._batch_ids_queue.put(ids)

  def _fill_feed_queue(self):
    while not self._coord.should_stop():
      batch_ids = self._batch_ids_queue.get()
      if batch_ids is False:
        self._feed_queue.put(False)
        break
      data = self._get_data(batch_ids)
      self._feed_queue.put(data)

  def _next_batch_ids(self):
    self._batch_index += 1
    if self._batch_index == len(self._batches):
      if not self._phase_train:
        return False
      self._reset_batches()
      self._batch_index = 0

    batch_start, batch_end = self._batches[self._batch_index]
    return self._shuffle_idx[batch_start: batch_end]

  def _reset_batches(self):
    if self._phase_train:
      if self._pos_neg_sample:
        self._shuffle_idx = []
        random.shuffle(self._vid_pos)
        random.shuffle(self._vid_neg)
        for b in xrange(self._nb_batch):
          for i in xrange(self._pos_num):
            self._shuffle_idx.append(
                self._all_vnames.index(
                    self._vid_pos[(b * self._pos_num  + i) % len(self._vid_pos)]))
          for i in xrange(self._neg_num):
            self._shuffle_idx.append(
                self._all_vnames.index(
                    self._vid_neg[(b * self._neg_num  + i) % len(self._vid_neg)]))
      else:
        np.random.shuffle(self._shuffle_idx)
    self._batch_index = -1

  def next(self):
    self._data_status.update()
    return self._feed_queue.get(timeout=None)

  def steps_per_epoch(self):
    return int(np.ceil(self._num_instances / float(self._batch_size)))

  def get_video_buckets(self):
    self._bucket_maps = []
    for info in self._meta_info:
      name = info['name']
      handle = self._name2split[name]
      data = handle[name]
      slot_match = False
      for bucket in self._buckets:
        if data.shape[0] < bucket[0]:
          self._bucket_maps[bucket[0]].append(name)
          slot_match = True
          break
      if not slot_match:
        self._bucket_maps[-1].append(name)

class DataStatus(object):
  def __init__(self, batch_size, n_samples):
    self._step = 0
    self._epoch = 0
    self._current_iter = 0
    self._batch_size = batch_size
    self._n_samples = n_samples

  def update(self):
    self._step += 1
    self._current_iter = min(self._step * self._batch_size, self._n_samples)

    if self._current_iter == self._n_samples:
      self._epoch += 1
      self._step = 0

  def reset(self):
    self._step = 0
    self._epoch = 0
