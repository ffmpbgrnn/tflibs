import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# dataset settings
tf.app.flags.DEFINE_string('visual_feature_dir', '',
                           """input features dir""")
# Definition of the pkl file
'''
Path to a pkl file.
train/
  name: 'image_name/video_name'
  other_attributes (e.g., label): 10
val/
test/
'''
tf.app.flags.DEFINE_string('dataset_path', '',
                           """dataset definition""")
tf.app.flags.DEFINE_string('split_name', '', """split_name""")
tf.app.flags.DEFINE_integer('max_queue_size', 10,
                            """input queue size""")
tf.app.flags.DEFINE_integer('num_loaders', 2,
                           """num thread of loading""")
tf.app.flags.DEFINE_string('stage', '',
                            """train/val/test""")
tf.app.flags.DEFINE_string('input_random_type', 'random_sample',
                           """random sample inputs""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.5, """gpu fraction""")
tf.app.flags.DEFINE_string('feature_extraction_dir', '',
                           """feature extraction dir""")
tf.app.flags.DEFINE_string('feat_save_path', '',
                           """feature save path""")
tf.app.flags.DEFINE_string('train_dir', '',
                           """train dir""")
tf.app.flags.DEFINE_float('num_ckpt_per_epoch', 1,
                           """number of models to save per epoch""")
tf.app.flags.DEFINE_boolean('random_input_order', False,
                          """random input""")

# Model settings
tf.app.flags.DEFINE_integer('fea_pool_size', 1,
                            """pool length""")
tf.app.flags.DEFINE_integer('fea_vec_size', 1, """vector size""")
tf.app.flags.DEFINE_integer('enc_out_size', 512, """encoder output size""")

# Train settings
tf.app.flags.DEFINE_float('lr', 0.001,
                            """learning rate""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """weight decay""")
tf.app.flags.DEFINE_integer('step_per_print', 10,
                            """print loss step""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.999,
                          """print loss step""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                          """batch size""")
tf.app.flags.DEFINE_boolean('phase_train', True,
                          """phase train""")
tf.app.flags.DEFINE_boolean('do_cls', False,
                          """whether to cls""")
tf.app.flags.DEFINE_boolean('use_sigmoid', False,
                          """use sigmoid or softmax""")
tf.app.flags.DEFINE_integer('num_answer_candidates', 0,
                          """num of answer candidats""")
tf.app.flags.DEFINE_integer('max_gradient_norm', 5,
                          """max gradient norm""")
# Eval settings
tf.app.flags.DEFINE_string('model_ckpt_path', "",
                          """model ckpt""")
tf.app.flags.DEFINE_string('load_model_path', "",
                          """load model path""")
tf.app.flags.DEFINE_string('label_info_path', "",
                          """label info pkl""")
tf.app.flags.DEFINE_integer('cls_num_classes', 0,
                          """num of classes""")

# Task settings
tf.app.flags.DEFINE_string('io_module', '', """io loader""")
tf.app.flags.DEFINE_string('nn_task', '', """this task""")

# Skipthoughts
tf.app.flags.DEFINE_integer('max_whole_seq_len', 250, "max video length")
tf.app.flags.DEFINE_integer('enc_cell_size', 0, "encoder cell size")
tf.app.flags.DEFINE_integer('dec_cell_size', 0, "decoder cell size")
tf.app.flags.DEFINE_integer('enc_seq_len', 0, "encoder sequence length")
tf.app.flags.DEFINE_integer('dec_seq_len', 0, "decoder sequence length")
tf.app.flags.DEFINE_integer('enc_num_layers', 0, "encoder num layers")
tf.app.flags.DEFINE_integer('dec_num_layers', 0, "decoder num layers")
tf.app.flags.DEFINE_integer('num_decoders', 1, "num decoders")
tf.app.flags.DEFINE_boolean('input_reverse', True, "reverse input")

# MemNN
tf.app.flags.DEFINE_string('kb_feature_dir', '', """kb feature dir""")
tf.app.flags.DEFINE_string('kb_meta_path', '', """kb label info path""")

tf.app.flags.DEFINE_integer('num_mem_slots', 1, "number of mem slots")
tf.app.flags.DEFINE_integer('num_hops', 1, "number of searching hops")

tf.app.flags.DEFINE_integer('key_vec_size', 0, "key representation size")
tf.app.flags.DEFINE_integer('key_emb_size', 0, "key embedded size")

tf.app.flags.DEFINE_integer('val_vocab_size', 0, "value vocaburary size")
tf.app.flags.DEFINE_integer('val_emb_size', 0, "value embedded size")

tf.app.flags.DEFINE_integer('query_vec_size', 0, "query representation size")
tf.app.flags.DEFINE_integer('query_emb_size', 0, "query embedded size")
tf.app.flags.DEFINE_integer('query_vocab_size', 0, "query vocaburary size")
tf.app.flags.DEFINE_string('embedding_weight_path', '', "path of embedding weight")


# Captioning
# Constants
tf.app.flags.DEFINE_integer('PAD_ID', 0, "PAD")
tf.app.flags.DEFINE_integer('GO_ID', 1, "GO")
tf.app.flags.DEFINE_integer('EOS_ID', 2, "EOS")
tf.app.flags.DEFINE_integer('UNK_ID', 3, "UNK")
tf.app.flags.DEFINE_string('coco_eval_dir', '', "py coco eval path")
tf.app.flags.DEFINE_integer('target_vocab_size', 0, "target vocabulary size")
tf.app.flags.DEFINE_integer('dec_embedding_size', 0, "decoder embedding size")
tf.app.flags.DEFINE_integer('attention_vec_size', 0, "attention vec size")
