import tensorflow as tf
from ltf.models.base import NNModel
from ltf.lib.rnn import rnn
from ltf.lib.rnn import rnn_cell
from ltf.lib.rnn import seq2seq
from ltf.models.captioning import encoder
from ltf.models.captioning import beam_search
from beam_search import Caption
from beam_search import TopN
import numpy as np
from ltf.models.skipthoughts import enc_dec
import os
import math

FLAGS = tf.app.flags.FLAGS

class Captioning(NNModel):
  def _get_dec_cell(self, cell_size, num_layers):
    single_cell = rnn_cell.GRUCell(cell_size)
    if self._phase_train:
      single_cell = rnn_cell.DropoutWrapper(
          single_cell, output_keep_prob=0.5, input_keep_prob=0.5)
    cell = single_cell
    if num_layers > 1:
      cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
    return cell

  def __init__(self):
    super(Captioning, self).__init__()
    self._batch_size = FLAGS.batch_size
    self._enc_seq_len = FLAGS.enc_seq_len
    self._dec_seq_len = FLAGS.dec_seq_len
    self._fea_pool_size = FLAGS.fea_pool_size
    self._fea_vec_size = FLAGS.fea_vec_size
    self._target_vocab_size = FLAGS.target_vocab_size
    self._forward_only = not FLAGS.phase_train
    self._phase_train = FLAGS.phase_train
    self._dec_cell_size = FLAGS.dec_cell_size
    self._enc_cell_size = FLAGS.enc_cell_size
    self._enc_out_size = FLAGS.fea_vec_size
    self._enc_num_layers = FLAGS.enc_num_layers
    self._dec_num_layers = FLAGS.dec_num_layers
    self._dec_embedding_size = FLAGS.dec_embedding_size
    self._num_heads = 1
    self._caption_outs = {}
    self._enc_use_clockwork = True
    self._dec_use_clockwork = False
    self._use_autoencoder = False
    self._use_lstm = False
    self._use_bs = True

    ## BS args
    self.beam_size = 3
    self.max_caption_length = self._dec_seq_len
    self.length_normalization_factor = 0.0

    self._enc_raw_inputs, self._enc_inputs, self._dec_inputs = [], [], []
    self.target_weights = []
    self.dec_cell = self._get_dec_cell(self._dec_cell_size, self._dec_num_layers)
    for i in xrange(self._enc_seq_len):
      self._enc_raw_inputs.append(tf.placeholder(
          tf.float32,
          shape=[None, self._fea_pool_size, self._fea_vec_size],
          name="encoder{0}".format(i)))
      self._enc_inputs.append(tf.reshape(self._enc_raw_inputs[-1], [-1, self._fea_vec_size]))
    for i in xrange(self._dec_seq_len):
      self._dec_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    targets = [self._dec_inputs[i + 1]
               for i in xrange(len(self._dec_inputs) - 1)]
    targets += [tf.zeros_like(self._dec_inputs[0])]

    output_projection = None
    softmax_loss_function = None

    # Decoder.
    output_size = None
    if output_projection is None:
      output_size = self._target_vocab_size

    with tf.variable_scope('encoder_disc'):
      enc_cell = enc_dec._get_cell(self, self._enc_cell_size, self._enc_num_layers, scope="Encoder_Layer_0")
      self._enc_outputs, self._enc_state, _ = enc_dec.go_encoder(
          self, enc_cell, self._enc_inputs, None)
    self._restore_vars = tf.all_variables()

    if isinstance(self._forward_only, bool):
      states = [tf.reshape(e, [-1, 1, self._enc_out_size])
                    for e in self._enc_outputs]
      self._attention_states = tf.concat(1, states)
      if self._phase_train or not self._use_bs:
        self.outputs, _ = seq2seq.embedding_attention_decoder(
            self._dec_inputs, self._enc_state, self._attention_states, self.dec_cell,
            self._target_vocab_size, self._dec_embedding_size, num_heads=self._num_heads,
            output_size=output_size, output_projection=output_projection,
            feed_previous=self._forward_only,
            initial_state_attention=False)
      else:
        self.bs_dec_inputs_pl = [tf.placeholder(tf.int32, shape=[None], name="bs_dec_input")]
        self.bs_enc_state_pl = tf.placeholder(tf.float32, shape=[None, self._dec_cell_size], name="bs_enc_state")
        self.bs_attention_states_pl = tf.placeholder(tf.float32,
                                                     shape=[None, self._enc_seq_len, self._enc_out_size],
                                                     name="bs_attention_states")
        self.bs_softmax, self.bs_new_states = seq2seq.embedding_attention_decoder(
            self.bs_dec_inputs_pl, self.bs_enc_state_pl, self.bs_attention_states_pl, self.dec_cell,
            self._target_vocab_size, self._dec_embedding_size, num_heads=self._num_heads,
            output_size=self._target_vocab_size, output_projection=None,
            feed_previous=self._forward_only,
            initial_state_attention=False)
        self.bs_softmax = tf.nn.softmax(self.bs_softmax[0])

        tf.get_variable_scope().reuse_variables()
        self.bs_softmax1, self.bs_new_states1 = seq2seq.embedding_attention_decoder(
            self.bs_dec_inputs_pl, self.bs_enc_state_pl, self.bs_attention_states_pl, self.dec_cell,
            self._target_vocab_size, self._dec_embedding_size, num_heads=self._num_heads,
            output_size=self._target_vocab_size, output_projection=None,
            feed_previous=self._forward_only,
            initial_state_attention=True)
        self.bs_softmax1 = tf.nn.softmax(self.bs_softmax1[0])

    if self._phase_train:
      self._total_loss = seq2seq.sequence_loss(
          self.outputs, targets, self.target_weights,
          softmax_loss_function=softmax_loss_function)
      self._get_train_op(clip_norm='global')


  def _get_input_feed_dict(self, batch):
    input_feed = {}
    for l in xrange(self._enc_seq_len):
      input_feed[self._enc_raw_inputs[l].name] = batch['inputs'][l]
    if self._phase_train:
      _dec_inputs = batch['decoder_inputs']
      target_weights = batch['target_weights']
    else:
      _dec_inputs, target_weights = [], []
      _dec_inputs.append([FLAGS.GO_ID])
      target_weights.append([0.])
      for i in xrange(self._dec_seq_len - 1):
        _dec_inputs.append([FLAGS.PAD_ID])
        target_weights.append([0.])
    for l in xrange(self._dec_seq_len):
      input_feed[self._dec_inputs[l].name] = _dec_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    return input_feed

  def train_step(self, sess, batch, with_train_op=False):
    input_feed = self._get_input_feed_dict(batch)
    if with_train_op:
      op = self._train_op
      output_feed = [op, self._total_loss, self._cls_loss, self._reconstruct_loss, self._global_norm]
      loss = sess.run(output_feed, input_feed)
      return loss[1:]
    else:
      op = self._summary_op
      output_feed = [op, self._total_loss]
      _str, _ = sess.run(output_feed, input_feed)
      return _str

  def eval_step(self, sess, batch, fout=None):
    if self._use_bs:
      outputs = self.eval_step_bs(sess, batch, fout=None)
    else:
      outputs = self.eval_step_std(sess, batch, fout=None)
    vname = batch['names'][0]
    captions = []
    for o in outputs:
      if self._use_bs:
        captions = [batch['vocab'][w] for w in o.sentence[1:-1]]
        break
      else:
        captions.append(batch['vocab'][o])
    print("{0}: {1}".format(vname, " ".join(captions)))
    self._caption_outs[vname] = captions

  def eval_step_std(self, sess, batch, fout=None):
    input_feed = self._get_input_feed_dict(batch)
    output_feed = []
    for l in xrange(self._dec_seq_len):  # Output logits.
      output_feed.append(self.outputs[l])
    outputs = sess.run(output_feed, input_feed)

    outputs = [int(np.argmax(logit, axis=1)) for logit in outputs]
    if FLAGS.EOS_ID in outputs:
      outputs = outputs[:outputs.index(FLAGS.EOS_ID)]
    return outputs

  def eval_step_bs(self, sess, batch, fout=None):
    input_feed = self._get_input_feed_dict(batch)
    enc_state, attention_states = sess.run([self._enc_state, self._attention_states],
                                           input_feed)
    initial_beam = Caption(
        sentence=[FLAGS.GO_ID],
        state=enc_state[0],
        logprob=0.0,
        score=0.0,
        metadata=[""])
    partial_captions = TopN(self.beam_size)
    partial_captions.push(initial_beam)
    complete_captions = TopN(self.beam_size)

    def step(inputs, states, attns, cap_idx):
      input_feed = {}
      input_feed[self.bs_dec_inputs_pl[0].name] = inputs
      input_feed[self.bs_enc_state_pl.name] = states
      input_feed[self.bs_attention_states_pl.name] = attns
      if cap_idx == 0:
        return sess.run([self.bs_softmax, self.bs_new_states], feed_dict=input_feed)
      else:
        return sess.run([self.bs_softmax1, self.bs_new_states1], feed_dict=input_feed)

    for cap_idx in range(self.max_caption_length - 1):
      partial_captions_list = partial_captions.extract()
      partial_captions.reset()
      input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
      state_feed = np.array([c.state for c in partial_captions_list])
      softmax, new_states = step(input_feed, state_feed, attention_states, cap_idx)
      # softmax, new_states = self.model.inference_step(sess,
                                                      # input_feed,
                                                      # state_feed)
      metadata = None
      for i, partial_caption in enumerate(partial_captions_list):
        word_probabilities = softmax[i]
        state = new_states[i]
        # For this partial caption, get the beam_size most probable next words.
        words_and_probs = list(enumerate(word_probabilities))
        words_and_probs.sort(key=lambda x: -x[1])
        words_and_probs = words_and_probs[0:self.beam_size]
        # Each next word gives a new partial caption.
        for w, p in words_and_probs:
          if p < 1e-12:
            continue  # Avoid log(0).
          sentence = partial_caption.sentence + [w]
          logprob = partial_caption.logprob + math.log(p)
          score = logprob
          if metadata:
            metadata_list = partial_caption.metadata + [metadata[i]]
          else:
            metadata_list = None
          if w == FLAGS.EOS_ID:
            if self.length_normalization_factor > 0:
              score /= len(sentence)**self.length_normalization_factor
            beam = Caption(sentence, state, logprob, score, metadata_list)
            complete_captions.push(beam)
          else:
            beam = Caption(sentence, state, logprob, score, metadata_list)
            partial_captions.push(beam)
      if partial_captions.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions.size():
      complete_captions = partial_captions

    return complete_captions.extract(sort=True)

  def write_feat(self):
    import ltf.classifier.captioning_metrics as metric
    gt, res = metric.load_captions(self._caption_outs, FLAGS.dataset_path, FLAGS.stage)
    assert len(gt.keys()) == len(res.keys())
    ids = gt.keys()
    model_id = FLAGS.feature_extraction_dir.split('/')[-1]
    step_id = FLAGS.feat_save_path.split('/')[-1]
    model_dir = '/tmp/cap_scores/{0}'.format(model_id)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    log_out = open(os.path.join(model_dir, step_id), 'w')
    metric.score(gt, res, ids, log_out)
