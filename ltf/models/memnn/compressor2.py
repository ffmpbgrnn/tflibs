import tensorflow as tf
from ltf.models.memnn import memory
from ltf.models.memnn import utils
from tensorflow.python.ops import rnn_cell_impl

# pylint: disable=protected-access
_linear = rnn_cell_impl._linear
# pylint: enable=protected-access

def compressor(self, query):
  if not self._phase_train:
    num_slots = self._num_hidden_cats * self._batch_size
    image_memory = tf.reshape(self.image_memory, [1, -1, 1, self._img_cats_mem_size])
    descs_memory = tf.reshape(self.descs_memory, [1, -1, self._desc_cats_mem_size])
    return image_memory, descs_memory, 0

  external_key_states = tf.reshape(self._embedded_key_bank,
                                    [-1, self._num_slots, self._key_vec_size])
  external_val_states = self._embedded_val_bank
  external_key_mem = memory.Memory(self._key_vec_size, self._num_lstm_units,
                                   'compressor_key_memory', enable_query=True)
  external_key_mem.set_memory(external_key_states)

  builtin_key_mem = memory.Memory(self._img_cats_mem_size, self._num_lstm_units,
                                  'searching_key_memory', enable_query=True)
  builtin_key_mem.set_memory(
      tf.reshape(self.image_memory, [-1, self._num_hidden_cats, self._img_cats_mem_size]))

  builtin_val_mem = memory.Memory(self._desc_cats_mem_size, self._num_lstm_units,
                                  'searching_val_memory', enable_query=False)
  builtin_val_mem.set_memory(
      tf.reshape(self.descs_memory, [-1, self._num_hidden_cats, self._desc_cats_mem_size]))


  read_cell = utils.get_cell(self, out_proj=True)
  read_state = read_cell.zero_state(self.run_time_batch_size, dtype=tf.float32)
  ext_key_read_outs = external_key_mem.zero_memory(self.run_time_batch_size)
  search_key_read_outs = builtin_key_mem.zero_memory(self.run_time_batch_size)

  for x in xrange(5):
    if x > 0:
      tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("read_LSTM"):
      inp = _linear([query] + ext_key_read_outs + search_key_read_outs, self._num_lstm_units, True) # , scope="input_proj")
      read_out, read_state = read_cell(inp, read_state)
      ext_key_read_outs, read_out_weights, read_out_gates = external_key_mem.query(read_out)
      search_key_read_outs, _, search_write_gates = builtin_key_mem.query(read_out)

      ext_key_read_out = tf.concat(axis=0, values=ext_key_read_outs)
      ext_val_read_outs = utils.value_x_weight(external_val_states, read_out_weights)
      ext_val_read_out = tf.concat(axis=0, values=ext_val_read_outs)

    with tf.variable_scope("write_key"):
      builtin_key_mem.write(search_write_gates[0], ext_key_read_out)
    with tf.variable_scope("write_val"):
      builtin_val_mem.write(search_write_gates[0], ext_val_read_out)

  search_key_bank = tf.reshape(builtin_key_mem.get_memory(),
                                [-1, self._num_hidden_cats, 1, self._img_cats_mem_size])
  search_val_bank = builtin_val_mem.get_memory()
  search_key_bank = tf.reshape(search_key_bank, [1, -1, 1, self._img_cats_mem_size])
  search_val_bank = tf.reshape(search_val_bank, [1, -1, self._desc_cats_mem_size])
  return search_key_bank, search_val_bank, 0
