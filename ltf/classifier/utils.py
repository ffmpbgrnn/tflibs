import numpy as np
import time

def get_ap(scores, Y_labels, label_id=1, one_hot_label=False):
  row = scores
  if True:
    true_pos = 0
    precision = 0.
    pred = np.argsort(row)[::-1]
    for idx in xrange(Y_labels.shape[0]):
      max_idx = pred[idx]
      if one_hot_label:
        gnd = Y_labels[max_idx][label_id]
        gnd_value = 1
      else:
        gnd = Y_labels[max_idx]
        gnd_value = label_id
      if gnd == gnd_value:
        true_pos += 1
        precision += 1. * true_pos / (idx + 1)
    if true_pos > 0:
      precision /= true_pos
  return precision

def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

def get_ap_voc2007(scores, Y_labels, label_id=0, one_hot_label=False):
  nd = scores.shape[0]
  tp = np.zeros(nd)
  fp = np.zeros(nd)
  pred = np.argsort(scores)[::-1]
  ## Slow version
  if False:
    npos = 0
    for d in xrange(nd):
      if Y_labels[d][label_id] == 1:
        npos += 1
      max_idx = pred[d]
      if Y_labels[max_idx][label_id] == 1:
        tp[d] = 1.
      else:
        fp[d] = 1.
  else:
    npos = np.sum(Y_labels[:, label_id])
    y = Y_labels[pred, label_id]
    tp[np.where(y == 1)[0]] = 1.
    fp[np.where(y != 1)[0]] = 1.
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric=True)
  return ap
