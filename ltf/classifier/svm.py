import h5py
import sys
import numpy as np
from sklearn import svm
import cPickle as pkl
from ltf.classifier import utils
import os

meta_path = sys.argv[1]
feature_path = sys.argv[2]
if len(sys.argv) == 3:
  C = 1.
else:
  C = float(sys.argv[3])
fin = h5py.File(feature_path)
log_fpath = "{0}.result.C{1}".format(feature_path, C)
if os.path.exists(log_fpath):
  exit(0)
log_fout = open(log_fpath, 'w')
fea_names = fin.keys()
fea_names_dict = {}
for k in fea_names:
  fea_names_dict[k] = True
meta = pkl.load(open(meta_path))
load_test_feas = True
mean_ap = 0

for class_id in meta.keys():
  details = meta[class_id]
  log_str = "{0}: {1}".format(class_id, details['event_name'][0])
  log_fout.write(log_str+'\n')

  if load_test_feas:
    feas = []
    cnt = 0
    for idx, vid in enumerate(details['test_vids']):
      if not fea_names_dict.get(vid, False):
        print(vid)
        continue
      d = fin[vid].value
      cnt += 1
      d = d / np.linalg.norm(d)
      feas.append(d[None, :])
    test_feas = np.vstack(feas)
    load_test_feas = False

  train_feas = []
  train_labels = []
  train_not_in = 0
  for idx, vid in enumerate(details['train_vids']):
    if not fea_names_dict.get(vid, False):
      print(vid)
      if details['train_label'][idx] == 1:
        pass
      continue

    d = fin[vid].value
    d = d / np.linalg.norm(d)
    train_feas.append(d[None, :])
    train_labels.append(details['train_label'][idx])
  train_feas = np.vstack(train_feas)
  train_labels = np.asarray(train_labels)

  test_labels = []
  test_not_in = 0
  for idx, vid in enumerate(details['test_vids']):
    if not fea_names_dict.get(vid, False):
      test_not_in += 1
      if details['test_label'][idx] == 1:
        pass
      continue
    test_labels.append(details['test_label'][idx])
  clf = svm.LinearSVC(C=C,
      verbose=False,
      max_iter=10000,
      # kernel='linear',)
      # penalty='l2', loss='hinge',
      dual=True)
  clf.fit(train_feas, train_labels)
  scores = clf.decision_function(test_feas)

  ap = utils.get_ap(scores, np.array(test_labels))
  mean_ap += ap
  log_str = 'ap: {0}'.format(ap)
  log_fout.write(log_str+'\n')
log_str = 'map: {0}'.format(mean_ap / len(meta.keys()))
log_fout.write(log_str+'\n')
log_fout.flush()
