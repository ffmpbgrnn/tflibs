import tensorflow as tf
from ltf.models.memnn import similarity
from ltf.models.memnn import utils
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import rnn_cell_impl

# pylint: disable=protected-access
_linear = rnn_cell_impl._linear
# pylint: enable=protected-access

slim = tf.contrib.slim


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  def loop_function(prev, _):
    prev_symbol = tf.argmax(prev, 1)
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


def reader(self, query, key_bank, val_bank, num_slots=0):
  tile = True
  if tile:
    key_bank3d = tf.reshape(key_bank, [1, -1, self._img_cats_mem_size])
    key_bank3d = tf.tile(key_bank3d, [self.run_time_batch_size, 1, 1])
  else:
    key_bank3d = tf.reshape(key_bank, [-1, num_slots, self._img_cats_mem_size])
  query_size = key_bank3d.get_shape().as_list()[2]

  cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self._num_lstm_units)
  if self._phase_train:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=self._dropout_keep_prob,
        output_keep_prob=self._dropout_keep_prob)
  cell = core_rnn_cell.OutputProjectionWrapper(cell, self._num_lstm_units)

  state = cell.zero_state(self.run_time_batch_size, dtype=tf.float32)

  num_read_heads = 5
  rs, val_out = [], []
  for rid in xrange(num_read_heads):
    r = tf.zeros([self.run_time_batch_size, self._img_cats_mem_size], dtype=tf.float32)
    rs.append(r)
  for rid in xrange(num_read_heads):
    v = tf.zeros([self.run_time_batch_size, self._desc_cats_mem_size], dtype=tf.float32)
    val_out.append(v)

  num_steps = self._num_answer_candidates
  val_outs = []

  if tile:
    val_bank = tf.reshape(val_bank, [1, -1, self._desc_cats_mem_size])
    val_bank = tf.tile(val_bank, [self.run_time_batch_size, 1, 1])
  else:
    val_bank = tf.reshape(val_bank, [-1, num_slots, self._desc_cats_mem_size])
  num_slots = tf.shape(val_bank)[1]

  input_labels = tf.split(axis=1, num_or_size_splits=self._num_answer_candidates, value=self._visual_target_label)
  target_labels = tf.split(axis=1, num_or_size_splits=self._num_answer_candidates, value=self._target_labels)
  target_weights = tf.split(axis=1, num_or_size_splits=self._num_answer_candidates, value=self._labels_target_weight)
  losses, eval_score = [], []

  num_symbols = self._query_vocab_size
  embedding_matrix = tf.get_variable("embedding",
                              [num_symbols, self._num_lstm_units])

  prev = None
  loop_function = None
  if not self._phase_train:
    loop_function = _extract_argmax_and_embed(
        embedding_matrix, None, False)

  with tf.variable_scope('LSTM_reader'):
    for x in xrange(num_steps):
      if x > 0:
        tf.get_variable_scope().reuse_variables()
      inp = _linear([query] + rs, self._num_lstm_units, True)
      output, state = cell(inp, state)
      state_flattened = utils.flatten(state)

      rs, val_out = [], []
      for rid in xrange(num_read_heads):
        a = similarity.softmax_similarity(state_flattened, key_bank, query_size,
                                          scope="softmax_similarity_%d" % rid)
        r = tf.matmul(tf.expand_dims(a, 1), key_bank3d)
        r = tf.reshape(r, [-1, self._img_cats_mem_size])
        rs.append(r)

        a3d = tf.reshape(a, [-1, 1, num_slots])
        val_out.append(tf.reshape(tf.matmul(a3d, val_bank),
                                  [-1, self._desc_cats_mem_size]))

      q = query
      mem_out = tf.concat(axis=1, values=rs + val_out + [q])
      mem_out = slim.fully_connected(
          mem_out, 1024,
          activation_fn=tf.nn.relu,
          scope="fc0")
      mem_out = slim.dropout(mem_out, self._dropout_keep_prob, is_training=self._phase_train)

      logits = slim.fully_connected(
          mem_out, self._query_vocab_size,
          activation_fn=None,
          scope="target_W")
      if loop_function is not None:
        prev = logits

      if self._phase_train:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(target_labels[-1], [-1]))
        losses.append(loss)
      else:
        eval_score.append(tf.nn.softmax(logits))
    if self._phase_train:
      loss = tf.add_n(losses) / num_steps
      self._cls_loss = tf.reduce_mean(loss, name='cross_entropy')
    else:
      self.eval_score = eval_score
      self.eval_score = tf.add_n(eval_score)
