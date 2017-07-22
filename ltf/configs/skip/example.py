from ltf.utils import merge_two_dicts

med_year = 14
ekx = 100

dataset_path = 'MEDTest{0}/meta_EK{1}.pkl'.format(med_year, ekx) # TODO
visual_feature_dir = 'MED' # TODO

feature_extraction_dir = '' # TODO
train_dir = 'log/skip' # TODO
label_info_path = 'MED{0}-MEDTEST-EK{1}.pkl'.format(med_year, ekx) # TODO

use_sigmoid = False
if use_sigmoid:
  num_classes = 20
else:
  num_classes = 21

common_config = {
  'label_info_path': label_info_path,
  'do_cls': False,
  'train_dir': train_dir,
  'dataset_path': dataset_path,
  'split_name': 'MEDTest{0}'.format(med_year),
  'input_random_type': '',
  'fea_pool_size': 1,
  'fea_vec_size': 1024,
  'lr': 0.0001,
  'num_ckpt_per_epoch': 0.5,
  'moving_average_decay': 0.999,
  'io_module': 'visual_fc',
  'nn_task': 'skipthoughts',
  'step_per_print': 20,
  # 'enc_cell_size': 1500,
  'enc_cell_size': 1024,
  'dec_cell_size': 1024,
  'enc_out_size': 1024,
  'enc_seq_len': 30,
  'dec_seq_len': 30,
  'weight_decay': 0.0001,

  'num_decoders': 3,
  'enc_num_layers': 1,
  'dec_num_layers': 1,
  'input_reverse': True,
  'cls_num_classes': num_classes,
  'random_input_order': False,
  'use_sigmoid':  use_sigmoid
}

_train_config = {
  'max_queue_size': 500,
  'num_loaders': 10,
  'stage': 'train',
  'visual_feature_dir': visual_feature_dir,
  'batch_size': 32,
  'gpu_fraction': 0.95,
  'max_whole_seq_len': 90,
  'phase_train': True,
}

_cls_config = {
  'do_cls': True,
  'stage': 'cls_train',
}

_eval_config = {
  'max_queue_size': 5,
  'num_loaders': 5,
  'stage': 'cls_test',
  'visual_feature_dir': visual_feature_dir,
  'batch_size': 64,
  'gpu_fraction': 0.5,
  'phase_train': False,
  'feature_extraction_dir': feature_extraction_dir,
  'enc_seq_len': 150,
  'dec_seq_len': 150,
  'max_whole_seq_len': 300,
}

train_config = merge_two_dicts(common_config, _train_config)
eval_svm_config = merge_two_dicts(common_config, _eval_config)
eval_svm_config['stage'] = 'extract_feat'
eval_config = merge_two_dicts(common_config, _cls_config)
cls_eval_config = merge_two_dicts(eval_config, _eval_config)
# eval_config = cls_eval_config
eval_config = eval_svm_config
cls_train_config = merge_two_dicts(train_config, _cls_config)
