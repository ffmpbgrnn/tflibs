import tensorflow as tf
from tensorflow.python.util import nest

def _get_arg_size(args):
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]
  return args, total_arg_size

def _linear(args, bias, matrix, bias_term):
  args, _ = _get_arg_size(args)

  # Now the computation.
  # with tf.variable_scope(scope or "Linear"):
  # matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
  if len(args) == 1:
    res = tf.matmul(args[0], matrix)
  else:
    res = tf.matmul(tf.concat(1, args), matrix)
  if not bias:
    return res
  # bias_term = tf.get_variable(
      # "Bias", [output_size],
      # initializer=tf.constant_initializer(bias_start))
  return res + bias_term
