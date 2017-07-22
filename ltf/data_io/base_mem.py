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
from Queue import Queue as Queue
import threading
# import multiprocessing
# from multiprocessing import Queue
import tensorflow as tf
import os
import random

FLAGS = tf.app.flags.FLAGS

class DataLoader(object):
  def _set_num_instances(self):
    self._meta_info = self._dataset[FLAGS.stage]
    self._num_instances = len(self._meta_info)

  def __init__(self, coord):
    self._coord = coord
    with open(FLAGS.dataset_path) as fin:
      self._dataset = pkl.load(fin)

    self._phase_train = FLAGS.phase_train
    self._set_num_instances()
    self._batch_size = FLAGS.batch_size
    self._num_threads = FLAGS.num_loaders

    self._nb_batch = int(np.ceil(self._num_instances/ float(self._batch_size)))
    self._shuffle_idx = np.arange(self._num_instances)

    self._batches = [(i*self._batch_size, min(self._num_instances, (i+1) * self._batch_size))
            for i in range(0, self._nb_batch)]

    # Queue holding batch ids
    self._batch_ids_queue = Queue(FLAGS.max_queue_size * 5)
    self._feed_queue = Queue(FLAGS.max_queue_size)

    self._name2shards = {}
    self.map_shards(FLAGS.visual_feature_dir, self._meta_info, self._name2shards)

    self._data_status = DataStatus(self._batch_size, self._num_instances)
    self._data_status.reset()

    self._reset_batch_ptr = None
    self._reset_batches()

  def map_shards(self, feature_dir, meta_info, name2shards):
    feas_in_handles = []
    feas_in_names = []
    for n in os.listdir(feature_dir):
      feature_path = os.path.join(feature_dir, n)
      h = h5py.File(feature_path)
      feas_in_handles.append(h)
      d = {}
      for k in h.keys():
        d[k] = True
      feas_in_names.append(d)
    for instance in meta_info:
      if type(instance) == str:
        name = instance
      else:
        name = instance['name']
      found = False
      for p_idx, names in enumerate(feas_in_names):
        if names.get(name, False):
          name2shards[name] = feas_in_handles[p_idx]
          found = True
          break
      if not found:
        print("key %s not found." % name)
        sys.stdout.flush()
        os._exit(0)

  def start(self):
    bi_threads = [threading.Thread(target=self._fill_batch_ids_queue)]
    fd_threads = [threading.Thread(target=self._fill_feed_queue, args=(i,))
                  for i in range(self._num_threads)]
    # bi_threads = [multiprocessing.Process(target=self._fill_batch_ids_queue)]
    # fd_threads = [multiprocessing.Process(target=self._fill_feed_queue, args=(i,))
                  # for i in range(self._num_threads)]
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

  def _fill_feed_queue(self, thread_idx):
    while not self._coord.should_stop():
      batch_ids = self._batch_ids_queue.get()
      if batch_ids is False:
        self._feed_queue.put(False)
        break
      data = self._get_data(batch_ids, thread_idx)
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
      if self._reset_batch_ptr is None or self._reset_batch_ptr() is False:
        np.random.shuffle(self._shuffle_idx)

    self._batch_index = -1

  def next(self):
    self._data_status.update()
    return self._feed_queue.get(timeout=None)

  def steps_per_epoch(self):
    return int(np.ceil(self._num_instances / float(self._batch_size)))

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
