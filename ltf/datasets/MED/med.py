import cPickle as pkl
import numpy as np

class MED():
  def __init__(self, label_info_path, stage):
    self._label_info_path = label_info_path
    with open(self._label_info_path) as fin:
      self._label_info = pkl.load(fin)

    self._name_to_label = {}
    self._one_hot_multilabel_name_label = {}
    self._one_hot_name_label = {}
    self._label_to_names = {}
    self._num_classes = len(self._label_info.keys())
    cnt = 0
    # index 0 will be background
    event_id = 1
    pos_videos = []
    for event_name, event_info in self._label_info.iteritems():
      self._label_to_names[event_id] = []
      train_label = event_info['%s_label' % stage]
      for idx, vname in enumerate(event_info['%s_vids' % stage]):
        if self._one_hot_multilabel_name_label.get(vname, False) is False:
          self._one_hot_multilabel_name_label[vname] = np.zeros((self._num_classes * 2), dtype=np.int32)
          self._one_hot_name_label[vname] = np.zeros((self._num_classes), dtype=np.int32)

        if train_label[idx] == 1:
          self._one_hot_multilabel_name_label[vname][2 * (event_id - 1)] = 1
          self._one_hot_name_label[vname][event_id - 1] = 1

          self._name_to_label[vname] = event_id

          self._label_to_names[event_id].append(vname)
          pos_videos.append(vname)
          cnt += 1
        else:
          self._one_hot_multilabel_name_label[vname][2 * (event_id - 1) + 1] = 1

          if self._name_to_label.get(vname, -1) == -1:
            self._name_to_label[vname] = 0
      event_id += 1
    self._label_to_names[0] = list(set(self._name_to_label.keys()) - set(pos_videos))

  def get_label(self, is_one_hot=False):
    if is_one_hot:
      return self._one_hot_name_label
    else:
      return self._name_to_label
  def cats_to_names(self):
    return self._label_to_names
