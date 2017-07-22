import tensorflow as tf

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS


def simple_pooling(self, encoder_inputs):
  enc_seq_len = len(encoder_inputs)
  attns = []
  encoder_state = None
  for i in xrange(enc_seq_len):
    conv_pool = tf.reduce_mean(self.encoder_inputs[i], [1])
    if i > 0:
      reuse= True
    else:
      reuse = False

    if encoder_state is None:
      encoder_state = conv_pool
    else:
      encoder_state = encoder_state + conv_pool
    attns.append(tf.reshape(conv_pool, [-1, 1, self._fea_vec_size]))
  encoder_state /= enc_seq_len

  encoder_state = tf.concat(1, [encoder_state] * self._dec_num_layers)
  attention_states = tf.concat(1, attns)
  return encoder_state, attention_states

def simple_2_pooling(self, encoder_inputs):
  enc_seq_len = len(encoder_inputs)
  attns = []
  batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.9997,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
  }
  encoder_state = None
  for i in xrange(enc_seq_len):
    if i > 0:
      reuse = True
    else:
      reuse = False
    attn = tf.reshape(encoder_inputs[i],
        [-1, self.encoder_pool_size, self.encoder_pool_size, self.conv_channel_len])
    attn = slim.conv2d(attn, 512, (1, 1), stride=(1, 1),
                           batch_norm_params=batch_norm_params,
                           activation=tf.nn.relu, reuse=reuse, scope="reduce_dim_0")
    state = slim.avg_pool(attn,
        (self.encoder_pool_size, self.encoder_pool_size), stride=1, scope="state_pooling")
    if encoder_state is None:
      encoder_state = state
    else:
      encoder_state = tf.maximum(encoder_state, state)
    attn = slim.conv2d(attn, 256, (2, 2), stride=(2, 2),
                           batch_norm_params=batch_norm_params,
                           activation=None, reuse=reuse, scope="reduce_mean_1")
    attn = slim.avg_pool(attn, (3, 3), stride=2, scope="pool")
    attn = tf.reshape(attn, [-1, 1, 4, 256])
    attns.append(attn)
  encoder_state = tf.reshape(encoder_state, [-1, 512])
  encoder_state = tf.concat(1, [encoder_state] * self.dec_num_layers)
  attns = tf.concat(1, attns)
  if False:
    attention_states = tf.reshape(encoder_inputs,
        [-1, enc_seq_len * self.conv_pool_len, self.conv_channel_len])
  else:
    attention_states = tf.reshape(attns,
        [-1, enc_seq_len * 4, 256])
  return encoder_state, attention_states
