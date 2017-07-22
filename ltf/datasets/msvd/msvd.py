import cPickle as pkl
import re
import string
from parse import get_train_vids
from parse import get_val_vids
from parse import get_test_vids
from parse import get_descs


def prepro_captions(captions):
  for stage in captions.keys():
    caps = captions[stage]
    for cap in caps:
      cap['tokens'] = cap['desc'].lower().translate(None, string.punctuation).strip().split()

def build_vocab(captions, count_thr=3):
  counts = {}
  sent_lengths = {}
  for stage in captions.keys():
    caps = captions[stage]
    for cap in caps:
      cap['tokens'] = cap['desc'].lower().translate(None, string.punctuation).strip().split()
      nw = len(cap['tokens'])
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
      for w in cap['tokens']:
        counts[w] = counts.get(w, 0) + 1

  cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
  print 'top words and their counts:'
  print '\n'.join(map(str,cw[:100]))

  # print some stats
  total_words = sum(counts.itervalues())
  print 'total words:', total_words
  bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
  vocab = [w for w,n in counts.iteritems() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
  print 'number of words in vocab would be %d' % (len(vocab), )
  print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

  # lets look at the distribution of lengths as well

  max_len = max(sent_lengths.keys())
  print 'max length sentence in raw data: ', max_len
  print 'sentence length distribution (count, number of words):'
  sum_len = sum(sent_lengths.values())
  for i in xrange(max_len+1):
    print '%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    # vocab.append('UNK')

  for stage in captions.keys():
    caps = captions[stage]
    for cap in caps:
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in cap['tokens']]
      cap['final_captions'] = caption
  vocab = ['PAD', 'GO', 'EOS', 'UNK'] + vocab
  captions['vocab'] = vocab

descs = get_descs()
captions = {}
dataset = 'msvd'
if dataset == 'msvd':
  mapping = {}
  with open('./youtube_mapping.txt') as fin:
    for line in fin.readlines():
      line = line.strip()
      line = line.split()
      mapping[line[1]] = line[0]
  captions['train'], captions['validate'] = [], []
  captions['test'] = []
  def create_input_format(vids, stage):
    for vid in vids:
      # mapped_vid = mapping[vid]
      mapped_vid = vid
      for d in descs[mapped_vid]:
        d['vname'] = vid
        captions[stage].append(d)
  create_input_format(get_train_vids(), 'train')
  create_input_format(get_val_vids(), 'validate')
  create_input_format(get_test_vids(), 'test')

count_thr = 1
UNK_ID = 3
build_vocab(captions, count_thr=count_thr)

for stage in captions.keys():
  if stage == 'train' or stage == 'test' or stage == 'validate':
    caps = captions[stage]
    for cap in caps:
      cap['caption_inputs'] = []
      for w in cap['final_captions']:
        if w == 'UNK':
          cap['caption_inputs'].append(UNK_ID)
        else:
          vocab_idx = captions['vocab'].index(w)
          cap['caption_inputs'].append(vocab_idx)
pkl.dump(captions, open('captions_msvd_{0}.pkl'.format(count_thr), 'w'))
