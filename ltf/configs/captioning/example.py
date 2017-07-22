from ltf.utils import merge_two_dicts

# vocab
# 0, 13023
# 1, 7174
# 2, 5506
# 3, 4604
dataset_path = 'captions_msvd_0.pkl' # TODO
visual_feature_dir = 'inception-v2_feas/5fps' # TODO
train_dir = '/tmp/log' # TODO
fea_vec_size = 1024 # TODO
feature_extraction_dir = 'cap_extracted' # TODO
coco_eval_dir = '' # TODO

common_config = {
  'do_cls': True,
  'train_dir': train_dir,
  'dataset_path': dataset_path,
  'fea_pool_size': 1,
  'fea_vec_size': fea_vec_size,
  'lr': 0.0001,
  'num_ckpt_per_epoch': 1,
  'moving_average_decay': 0.999,
  'io_module': 'visual_text',
  'nn_task': 'captioning',
  'step_per_print': 20,
  # 'enc_cell_size': 1500,
  'enc_cell_size': 1024,
  'dec_cell_size': 1024,
  'dec_embedding_size': 1024,
  'enc_seq_len': 50,
  'dec_seq_len': 15,
  'weight_decay': 0.00005,

  'enc_num_layers': 1,
  'dec_num_layers': 1,
  'max_gradient_norm': 5,
  'target_vocab_size': 13015,
  'attention_vec_size': 100,
}

_train_config = {
  'max_queue_size': 100,
  'num_loaders': 10,
  'stage': 'train',
  'visual_feature_dir': visual_feature_dir,
  'batch_size': 32,
  'gpu_fraction': 0.95,
  'phase_train': True,
}

_eval_config = {
  'max_queue_size': 5,
  'num_loaders': 5,
  'enc_seq_len': 150,
  'stage': 'test',
  'visual_feature_dir': visual_feature_dir,
  'batch_size': 1,
  'gpu_fraction': 0.5,
  'phase_train': False,
  'feature_extraction_dir': feature_extraction_dir,
  'coco_eval_dir': coco_eval_dir,
}

train_config = merge_two_dicts(common_config, _train_config)
eval_config = merge_two_dicts(common_config, _eval_config)
