import sys
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
LIBRARY_PATH = FLAGS.coco_eval_dir
sys.path.append(LIBRARY_PATH)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import cPickle as pkl


def score(gts, res, ids, log_out):
  tokenizer = PTBTokenizer()
  gts  = tokenizer.tokenize(gts)
  res = tokenizer.tokenize(res)
  scorers = [
      (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
      (Meteor(),"METEOR"),
      (Rouge(), "ROUGE_L"),
      (Cider(), "CIDEr")]
  for scorer, method in scorers:
    # print 'computing %s score...'%(scorer.method())
    score, scores = scorer.compute_score(gts, res)
    if type(method) == list:
      for sc, scs, m in zip(score, scores, method):
        print>>log_out, "%s: %f"%(m, sc)
    else:
        print>>log_out, "%s: %f"%(method, score)

def load_captions(generated_caption, caption_meta_path, stage):
  meta = pkl.load(open(caption_meta_path))
  gt, res = {}, {}
  for v in meta[stage]:
    vname = v['vname']
    if vname not in gt.keys():
      gt[vname] = []
    gt[vname].append({'caption': v['desc']})

  for vname, caption in generated_caption.iteritems():
    res[vname] = [{"caption": " ".join(caption)}]
  return gt, res
