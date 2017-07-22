from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
from datetime import datetime
import time
import numpy as np
from ltf.data_io import visual_fc
from ltf.data_io import visual_fc_img
from ltf.data_io import visual_fc_img_matching
from ltf.data_io import visual_text
from ltf.flags import flags
from ltf import utils
import h5py
import random

FLAGS = tf.app.flags.FLAGS

class Expr():
  def __init__(self):
    seed = 1234
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    self._step_per_print = FLAGS.step_per_print
    print('starting loader')
    self._start_loader()
    print('getting model')
    self._get_model()
    print('initial session')
    self._init_sess()
    print('initial done')

  def _init_sess(self):
    self._sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)))
    self._sess.run(tf.initialize_all_variables())

    self._saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
    if not os.path.exists(FLAGS.train_dir):
      os.makedirs(FLAGS.train_dir)
    self._summary_writer = tf.train.SummaryWriter(
      FLAGS.train_dir,
      graph=self._sess.graph)

  def _start_loader(self):
    coord = tf.train.Coordinator()
    if FLAGS.io_module == 'visual_fc':
      self._data_loader = visual_fc.VisualFC(coord)
    elif FLAGS.io_module == 'visual_fc_img':
      self._data_loader = visual_fc_img.VisualFC(coord)
    elif FLAGS.io_module == 'visual_fc_img_matching':
      self._data_loader = visual_fc_img_matching.VisualFC(coord)
    elif FLAGS.io_module == 'visual_text':
      self._data_loader = visual_text.VisualText(coord)
    else:
      utils.print_and_exit("{0} is not implemented".format(FLAGS.io_module))
    self._steps_per_epoch = self._data_loader.steps_per_epoch()
    self._data_loader.start()

  def _get_model(self):
    # get model
    if FLAGS.nn_task == 'captioning':
      from ltf.models.captioning import captioning
      self._model = captioning.Captioning()
    elif FLAGS.nn_task == 'skipthoughts':
      from ltf.models.skipthoughts import skipthoughts
      self._model = skipthoughts.SkipThoughts(
        enc_cell_size=FLAGS.enc_cell_size, dec_cell_size=FLAGS.dec_cell_size,
        enc_seq_len=FLAGS.enc_seq_len, dec_seq_len=FLAGS.dec_seq_len,
        enc_num_layers=FLAGS.enc_num_layers, dec_num_layers=FLAGS.dec_num_layers,
      )
    elif FLAGS.nn_task == 'memnn':
      print('doing memnn')
      from ltf.models.memnn import memnn
      self._model = memnn.MemNN()
    elif FLAGS.nn_task == 'matching':
      print('doing matching')
      from ltf.models.memnn import matching
      self._model = matching.Matching()
    elif FLAGS.nn_task == 'gan':
      print('doing infogan')
      from ltf.models.gan import infogan
      self._model = infogan.InfoGAN()
    else:
      utils.print_and_exit("{0} is not implemented".format(FLAGS.nn_task))

  def do_train(self):
    if len(FLAGS.model_ckpt_path) > 0:
      self._restore(FLAGS.model_ckpt_path)
    elif len(FLAGS.load_model_path) > 0:
      self._restore(FLAGS.load_model_path)
    duration, io_duration = 0, 0
    step = 0
    if FLAGS.phase_train:
      log_fout = open(os.path.join(FLAGS.train_dir, 'train.log'), 'w')
    print('start training')
    while True:
      start_time = time.time()
      batch = self._data_loader.next()
      io_duration = time.time() - start_time
      loss_value, cls_loss_val, recon_loss_val, global_norm, regular_loss_val = self._model.train_step(self._sess, batch, with_train_op=True)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      duration += time.time() - start_time

      step += 1
      if step % self._step_per_print == 0:
        examples_per_sec = FLAGS.batch_size / (float(duration) / self._step_per_print)
        format_str = ('%s: step %d.%d, loss = %.2f, cls_loss = %.2f, recon_loss = %.2f, global norm = %.2f, regular_loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch; IO: %.3f sec/batch)')
        log_str = format_str % (datetime.now(), step / self._steps_per_epoch, step % self._steps_per_epoch,
                                loss_value, cls_loss_val, recon_loss_val, global_norm, regular_loss_val, examples_per_sec, duration / self._step_per_print,
                                io_duration / self._step_per_print)
        print(log_str)
        log_fout.write(log_str+'\n')
        log_fout.flush()
        duration, io_duration = 0, 0

      if step % (self._steps_per_epoch / 10)  == 0:
        summary_str = self._model.train_step(self._sess, batch, with_train_op=False)
        self._summary_writer.add_summary(summary_str, step)
      if step % int(self._steps_per_epoch / FLAGS.num_ckpt_per_epoch) == 0:
        print('saving ckpt to %s' % FLAGS.train_dir)
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        self._saver.save(self._sess, checkpoint_path, global_step=step)

  def _restore(self, model_ckpt_path):
    if FLAGS.phase_train:
      if self._model._restore_vars is None:
        saver = tf.train.Saver(tf.all_variables())
      else:
        saver = tf.train.Saver(self._model._restore_vars)
    else:
      variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
    print("restore model from {0}".format(model_ckpt_path))
    saver.restore(self._sess, model_ckpt_path)

  def do_eval(self):
    self._restore(FLAGS.model_ckpt_path)
    step_id = FLAGS.feat_save_path.split('/')[-1]
    score_log_fout = open(os.path.join(FLAGS.feature_extraction_dir, step_id+'.score'), 'w')

    step = 0
    ended_thread = 0
    while True:
      batch = self._data_loader.next()
      if batch is False:
        ended_thread += 1
        if ended_thread == FLAGS.num_loaders:
          break
        continue
      self._model.eval_step(self._sess, batch, fout=None)
      step += 1
      if step % 10 == 0:
        print("{0}/{1}".format(step, self._steps_per_epoch))
      if step % 50 == 0 and False:
        self._model.eval_get_metric(score_log_fout)
    self._model.write_feat()
    self._model.eval_get_metric(score_log_fout)

def run():
  expr = Expr()
  if FLAGS.phase_train:
    expr.do_train()
  else:
    expr.do_eval()

def main(argv=None):
  with tf.Graph().as_default():
    tf.set_random_seed(1234)
    run()

if __name__ == '__main__':
  tf.app.run()
