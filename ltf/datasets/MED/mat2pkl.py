import scipy.io as sio
import cPickle as pkl
import sys

med_name = sys.argv[1]
mat = sio.loadmat('{0}.mat'.format(med_name))
train_list, test_list = [], []
for i in xrange(mat['vid_train'].shape[0]):
  train_list.append(mat['vid_train'][i][0][0])
for i in xrange(mat['vid_test'].shape[0]):
  test_list.append(mat['vid_test'][i][0][0])

num_events = mat['events_info'][0].shape[0]
result = {}
for event_idx in xrange(num_events):
  d = mat['events_info'][0][event_idx][0][0]
  event_id = d[0][0]
  o = {}
  o['event_name'] = d[1]

  train_vid_idx = d[2]
  train_vid_labels = d[3]
  train_names = []
  train_labels = []
  for train_id in train_vid_idx:
    train_names.append(train_list[train_id[0] - 1])
  o['train_vids'] = train_names
  for train_label_id in train_vid_labels:
    train_labels.append(train_label_id[0])
  o['train_label'] = train_labels
  o['test_label'] = []
  for test_label_id in d[4]:
    o['test_label'].append(test_label_id[0])
  o['test_vids'] = test_list
  result[event_id] = o
pkl.dump(result, open("{0}.pkl".format(med_name), 'w'))
