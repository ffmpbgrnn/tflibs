import cPickle as pkl
import numpy as np

class OpenImages():
  def __init__(self, label_info_path, phase_train=True):
    self._phase_train = phase_train
    self._cats_names = [
      '<PAD>', '<SOS>', '<EOS>']
    self._label_info_path = label_info_path
    with open(self._label_info_path) as fin:
      self._meta_info = pkl.load(fin)
    self._cats_names += self._meta_info['vocab']
    self._num_classes = len(self._cats_names)

    label_info = self._meta_info['label_info']

    self._name_to_labels = {}
    self._name_to_multi_labels = {}
    for name, labels in label_info.iteritems():
      label_idxs = []
      multi_label = np.zeros([self._num_classes], dtype=np.int32)
      label_idxs.append(self._cats_names.index('<SOS>'))
      for l in labels:
        label_idx = self._cats_names.index(l)
        multi_label[label_idx] = 1
        label_idxs.append(label_idx)
      label_idxs.append(self._cats_names.index('<EOS>'))
      self._name_to_labels[name] = label_idxs
      self._name_to_multi_labels[name] = multi_label

  def get_label(self, is_one_hot=False):
    if self._phase_train:
      return self._name_to_labels
    else:
      return self._name_to_multi_labels

  def get_pad_id(self):
    return self._cats_names.index('<PAD>')
