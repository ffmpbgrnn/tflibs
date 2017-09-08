from ltf.utils import merge_two_dicts
import os

root_dir = '/tmp/mem/'
train_dir = os.path.join(root_dir, 'few_shots/train_dir/')
if not os.path.exists(train_dir):
  os.makedirs(train_dir)
extraction_dir = os.path.join(root_dir, 'extracted')
if not os.path.exists(extraction_dir):
  os.makedirs(extraction_dir)

dataset_path = os.path.join(root_dir, 'few_shots/meta.pkl')
visual_feature_dir = os.path.join(root_dir, 'few_shots/feats/')
query_vocab_size = 100
train_stage = 'train_1'
eval_stage = 'test'

nn_task = 'memnn'
step_per_print = 20
batch_size = 1
load_model_path = ''
num_ckpt_per_epoch = 0.5
num_mem_slots = 1000

common_config = {
  'use_sigmoid': False,
  'num_answer_candidates': 1,
  'train_dir': train_dir,
  'do_cls': True,
  'dataset_path': dataset_path,
  'fea_pool_size': 1,
  'fea_vec_size': 2048,
  'lr': 0.0001,
  'num_ckpt_per_epoch': num_ckpt_per_epoch,
  'moving_average_decay': 0.999,
  'max_gradient_norm': 10,
  'io_module': 'visual_fc_img',
  'nn_task': nn_task,
  'step_per_print': step_per_print,
  'weight_decay': 0.0001,
  # 'kb_feature_dir': "/data/uts200/linchao/memNN/datasets/imagenet/feas/",
  'kb_feature_dir': os.path.join(root_dir, 'ext_knowledge/feats/'),
  # 'kb_feature_dir': "/data/uts200/linchao/memNN/datasets/meta/imagenet/all/normalized_feas/",
  # 'kb_meta_path': "/data/uts311/linchao/MemNN/src/tf_libs/ltf/datasets/ImageNet/kb_meta.pkl",
  'kb_meta_path': os.path.join(root_dir, 'ext_knowledge/kb_meta_all.pkl'),
  'embedding_weight_path': os.path.join(root_dir, 'ext_knowledge/imagenet_1k_emb_matrix.pkl'),
  'num_mem_slots': num_mem_slots,
  'num_hops': 1,
  'key_vec_size': 2048,
  'val_vocab_size': 1000,
  'query_vec_size': 2048,
  'key_emb_size':   256, # 512
  'val_emb_size':   256,
  'query_emb_size': 256, # 512
  'query_vocab_size': query_vocab_size,
}

_train_config = {
  'max_queue_size': 400,
  'load_model_path': load_model_path,
  'num_loaders': 10,
  'stage': train_stage,
  'visual_feature_dir': visual_feature_dir,
  'batch_size': batch_size,
  'gpu_fraction': 0.5,
  'phase_train': True,
}

_eval_config = {
  'max_queue_size': 5,
  'num_loaders': 5,
  'stage': eval_stage,
  'visual_feature_dir': visual_feature_dir,
  'batch_size': 128,
  'gpu_fraction': 0.5,
  'phase_train': False,
  'feature_extraction_dir': extraction_dir,
}

train_config = merge_two_dicts(common_config, _train_config)
eval_config = merge_two_dicts(common_config, _eval_config)
