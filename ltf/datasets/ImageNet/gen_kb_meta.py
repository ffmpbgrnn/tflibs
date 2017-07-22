import cPickle as pkl
import random

img_list_path = 'train.list'
example_per_cat = 10

meta_info = {}
meta_info['name_list'] = []
meta_info['vocab'] = {}

idx = 0
with open('imagenet_labels.txt') as fin:
  for line in fin.readlines():
    line = line.strip()
    meta_info['vocab'][line] = idx
    idx += 1

cat_names = {}
with open(img_list_path) as fin:
  for line in fin.readlines():
    line = line.strip()
    line = line.split('/')
    img_name = line[-1]
    img_name = img_name.split('.')[0]
    cat_name = line[-2]
    cat = cat_names.get(cat_name, [])
    cat.append(img_name)
    cat_names[cat_name] = cat

# do sample
for cat, img_names in cat_names.iteritems():
  random.shuffle(img_names)
  for img_name in img_names[: example_per_cat]:
    meta_info['name_list'].append(img_name)

pkl.dump(meta_info, open('kb_meta_all.pkl', 'w'))
