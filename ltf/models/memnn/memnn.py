import tensorflow as tf
from ltf.models.base import NNModel
import numpy as np
from ltf.models.memnn import compressor2
from ltf.models.memnn import reader
import cPickle as pkl


slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

class MemNN(NNModel):
  def __init__(self):
    super(MemNN, self).__init__()
    self._phase_train = FLAGS.phase_train
    self._batch_size = FLAGS.batch_size
    self._num_hops = FLAGS.num_hops
    self._num_slots = FLAGS.num_mem_slots
    self._key_vec_size, self._key_emb_size = FLAGS.key_vec_size, FLAGS.key_emb_size
    self._query_vocab_size = FLAGS.query_vocab_size

    self._val_vocab_size = FLAGS.val_vocab_size
    self._val_emb_size = FLAGS.val_emb_size

    self._query_vec_size = FLAGS.query_vec_size
    self._query_emb_size = FLAGS.query_emb_size

    self._score_dict = None
    self._use_sigmoid = FLAGS.use_sigmoid
    self._num_answer_candidates = FLAGS.num_answer_candidates

    # assert self._query_emb_size == self._key_emb_size

    self._visual_key_bank = tf.placeholder(
        tf.float32, shape=[None, self._num_slots, self._key_vec_size],
        name="visual_key")

    self._semantic_val_bank = tf.placeholder(
        tf.int32, shape=[None, self._num_slots],
        name="semantic_val")

    self._visual_query = tf.placeholder(
        tf.float32, shape=[None, self._query_vec_size],
        name="visual_input_rep")
    if self._use_sigmoid:
      self._visual_target_label = tf.placeholder(
          tf.float32, shape=[None, self._query_vocab_size],
          name="visual_target_label")
    else:
      self._visual_target_label = tf.placeholder(
          tf.int32, shape=[None, self._num_answer_candidates],
          name="visual_target_label")
    self._labels_target_weight = tf.placeholder(
          tf.float32, shape=[None, self._num_answer_candidates],
          name="visual_target_weights")
    self._target_labels = tf.placeholder(
          tf.int32, shape=[None, self._num_answer_candidates],
          name="target_labels")

    self._dropout_keep_prob = 0.5
    batch_norm_var_collection = 'moving_vars'
    self.batch_norm_params = {
        'is_training': self._phase_train,
        'decay': 0.97,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(0.0001)):
      self.build()

  def build(self):
    self._key_bank = tf.reshape(self._visual_key_bank, [-1, 1, self._num_slots, self._key_vec_size])
    self._embedded_key_bank = tf.reshape(self._key_bank,
                                         [-1, 1, self._num_slots, self._key_vec_size])

    with tf.variable_scope("val_embedding"):
      with open(FLAGS.embedding_weight_path) as fin:
        one_m_1b_w = pkl.load(fin)
      self._val_emb_matrix = tf.constant(
          one_m_1b_w, dtype=tf.float32, shape=[self._val_vocab_size, 1024], name="val_embedding_matrix")
      self._embedded_val_bank = tf.nn.embedding_lookup(self._val_emb_matrix, self._semantic_val_bank)
      _val_bank = tf.reshape(self._embedded_val_bank, [-1, self._num_slots, 1, 1024])
      self._embedded_val_bank = tf.reshape(_val_bank, [-1, self._num_slots, 1024])

    with tf.variable_scope("searching_memory"):
      self._num_hidden_cats = 100
      self._img_cats_mem_size = self._key_emb_size
      self._desc_cats_mem_size = self._val_emb_size
      self.image_memory = tf.get_variable("image_memory",
                                          [self._num_hidden_cats, self._img_cats_mem_size])
      self.descs_memory = tf.get_variable("descs_memory",
                                          [self._num_hidden_cats, self._desc_cats_mem_size])

    regression_loss = []
    self.run_time_batch_size = tf.shape(self._visual_query)[0]

    def step(query, i, step_id):
      self._num_lstm_units = 1024
      query = slim.dropout(query, self._dropout_keep_prob, is_training=self._phase_train)
      with tf.variable_scope('compressor_%d' % i):
        search_key_bank, search_val_bank, num_slots = compressor2.compressor(self, query)
        loss = tf.constant(0., dtype=tf.float32, name="regress_loss")
        regression_loss.append(loss)

      step_mode = True
      with tf.variable_scope('reader_%d' % i):
        reader.reader(self, query, search_key_bank, search_val_bank, num_slots)

      out = None
      if not step_mode:
        out = slim.fully_connected(
            read_out, self._query_vec_size,
            activation_fn=tf.nn.relu, scope="output_W_%d" % i)
        out = slim.dropout(out, self._dropout_keep_prob,
                          is_training=self._phase_train)
      return out

    query = self._visual_query
    share_param = True
    share_stride = 1
    with tf.variable_scope("addressing"):
      for step_id in xrange(self._num_hops):
        stride_id = step_id
        if step_id >= share_stride:
          if share_param:
            tf.get_variable_scope().reuse_variables()
            stride_id = step_id % share_stride
        query = step(query, stride_id , step_id)

    if self._phase_train:
      self._total_loss = self._cls_loss
      self._get_train_op('global')

  def _get_input_feed_dict(self, batch):
    input_feed = {}
    input_feed[self._visual_query.name] = batch['inputs']
    if self._phase_train:
      input_feed[self._visual_target_label.name] = batch['labels']
      input_feed[self._target_labels.name] = batch['target_labels']
      input_feed[self._labels_target_weight.name] = batch['target_weights']
      input_feed[self._visual_key_bank.name] = batch['kb_key']
      input_feed[self._semantic_val_bank.name] = batch['kb_val']
    # call gan.py->feed_input()
    return input_feed


  def train_step(self, sess, batch, with_train_op=True):
    input_feed = self._get_input_feed_dict(batch)
    if with_train_op:
      op = self._train_op
      output_feed = [op, self._total_loss, self._cls_loss, self._reconstruct_loss, self._global_norm,
                     self._regular_loss]
      loss = sess.run(output_feed, input_feed)
      return loss[1:]
    else:
      op = self._summary_op
      output_feed = [op, self._total_loss]
      _str, _ = sess.run(output_feed, input_feed)
      return _str

  def eval_step(self, sess, batch, fout=None):
    input_feed = self._get_input_feed_dict(batch)
    logits = sess.run(self.eval_score, input_feed)

    if self._score_dict is None:
      self._score_dict, self._label_dict = {}, {}
    for x in xrange(len(batch['names'])):
      name = batch['names'][x]
      self._score_dict[name] = logits[x]
      self._label_dict[name] = batch['labels'][x]

  def eval_get_metric(self, score_log):
    scores, labels = [], []
    for name in self._score_dict.keys():
      scores.append(self._score_dict[name])
      labels.append(self._label_dict[name])
    scores = np.vstack(scores)
    labels = np.array(labels)

    _map = 0.
    num_classes = scores.shape[1]

    for i in xrange(scores.shape[0]):
      pred_label = np.argmax(scores[i])
      gnd_label = np.argmax(labels[i])
      if pred_label == gnd_label:
        _map += 1
    _map /= scores.shape[0]

    print("mAP: {0}".format(_map))
    print>>score_log, _map
