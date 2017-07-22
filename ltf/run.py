import os
import sys
import cPickle as pkl
import subprocess
import argparse
import time


class Tasks(object):
  def __init__(self, args):
    self._gpu_ids = args.gpu
    self._config_path = args.config_path
    self._restore_path = args.restore_path
    self._task = args.task
    self._eval_model_id = args.model_id
    self._running_process = []

  def _run_me(self, cmds, wait=True):
    if len(self._running_process) > 50:
      for p in self._running_process:
        p.wait()
      self._running_process = []
    p = subprocess.Popen(cmds, shell=True)
    if wait:
      p.wait()
    else:
      self._running_process.append(p)

  def _gen_cmd_and_run(self, config, overwrite_feat=True):
    # create training commands
    # this_dir = "{0}/../".format(pwd)
    run_cmd = True
    if self._task == "eval":
      p = config['model_ckpt_path'].split('/')
      step_id, model_id = p[-1], p[-2]
      save_dir = os.path.join(config['feature_extraction_dir'], model_id)
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      save_step_path = os.path.join(save_dir, step_id)
      if os.path.exists(save_step_path) and not overwrite_feat:
        run_cmd = False
      else:
        rep_id = 0
        while os.path.exists(save_step_path):
          save_step_path = "{0}.{1}".format(save_step_path, rep_id)
          rep_id += 1
      config['feat_save_path'] = save_step_path
      config['feature_extraction_dir'] = save_dir

    if run_cmd:
      cmds = '''cd {0}; export PYTHONPATH="{0}/../"; export CUDA_VISIBLE_DEVICES={1}; python train.py'''.format(self._pwd, self._gpu_ids)
      for k, v in config.iteritems():
        cmds = "{0} \\\n--{1}={2}".format(cmds, k, v)
      print(cmds)
      self._run_me(cmds)

    # do evaluation, get map
    if self._task == "eval" and config['do_cls'] is False:
      pass

  def _get_max_model_id(self, train_dir):
    max_idx = -1
    for fname in os.listdir(train_dir):
      sub_dir = os.path.join(train_dir, fname)
      if os.path.isdir(sub_dir):
        idx = int(fname)
        if idx > max_idx:
          max_idx = idx
    return max_idx + 1

  def _eval_all(self, old_config):
    for eval_model_id in self._eval_model_id.split(','):
      train_dir = os.path.join(old_config['train_dir'], eval_model_id)
      self._pwd = os.path.join(train_dir, 'src', 'ltf')
      config_path = os.path.join(old_config['train_dir'], eval_model_id, 'src', 'ltf', self._config_path)
      eval_config = {}
      with open(config_path) as fin:
        exec(fin.read())
        config = eval_config
        for fname in os.listdir(train_dir):
          if "model.ckpt" in fname:
            names = fname.split('-')
            if len(names) == 2:
              try:
                int(names[1])
                self._task = "eval"
                config["model_ckpt_path"] = os.path.join(train_dir, fname)
                self._gen_cmd_and_run(config, overwrite_feat=False)
              except:
                pass

  def _show_best_result(self, result_dir):
    max_map = -1
    model_id = ""
    for fname in os.listdir(result_dir):
      if fname.split('.')[-1] == "0":
        result_path = os.path.join(result_dir, fname)
        with open(result_path) as fin:
          for line in fin.readlines():
            line = line.strip()
            if line[:3] == "map":
              _map = float(line.split(':')[-1].strip())
              if _map > max_map:
                max_map = _map
                model_id = fname
    print('max map: {0}; model_id: {1}'.format(max_map, model_id))

  def run(self):
    train_config, eval_config = {}, {}
    cls_train_config = {}
    with open(self._config_path) as fin:
      exec(fin.read())

      if self._task == "train":
        self._pwd = os.path.dirname(os.path.abspath(__file__))
      else:
        if self._task == "eval" or self._task == "continue" or self._task == 'do_cls':
          p = '/'.join(self._restore_path.split('/')[:-1])
          self._eval_model_id = p.split('/')[-1]
          self._config_path = os.path.join(
              train_config['train_dir'], self._eval_model_id, 'src', 'ltf', self._config_path)
        else:
          p = "/"
        self._pwd = os.path.join(p, 'src', 'ltf')

    with open(self._config_path) as fin:
      exec(fin.read())
      if self._task == "train" or self._task == 'continue' or self._task == 'do_cls':
        if self._task == "train" or self._task == 'continue':
          config = train_config
        elif self._task == 'do_cls':
          config = cls_train_config
        train_dir = config['train_dir']
        model_id = self._get_max_model_id(train_dir)
        print("Model id: {0}".format(str(model_id)))
        train_dir = os.path.join(train_dir, str(model_id))
        if not os.path.exists(train_dir):
          os.makedirs(train_dir)
          self._run_me("cp -ar {0}/../. {1}".format(self._pwd, os.path.join(train_dir, 'src')))

        config['train_dir'] = train_dir
        if self._task == 'continue' or self._task == 'do_cls':
          config["model_ckpt_path"] = self._restore_path
        self._gen_cmd_and_run(config)
      elif self._task == "eval" or self._task == 'eval_all_steps':
        config = eval_config
        if self._task == "eval":
          config["model_ckpt_path"] = self._restore_path
          self._gen_cmd_and_run(config)
        elif self._task == "eval_all_steps":
          while True:
            self._eval_all(config)
            time.sleep(5)
      elif self._task == "show_best":
        config = eval_config
        result_dir = os.path.join(config['feature_extraction_dir'], self._eval_model_id)
        self._show_best_result(result_dir)
      elif self._task == "list_result":
        config = eval_config
        result_dir = os.path.join(config['feature_extraction_dir'], self._eval_model_id)
        for res in os.listdir(result_dir):
          if "result" in res:
            statinfo = os.stat(os.path.join(result_dir, res))
            print("{0} {1}".format(res, statinfo.st_size))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Run me")
  parser.add_argument(
      '-g', '--gpu', required=False,
      default="0", help="GPU id")
  parser.add_argument(
      '-c', '--config_path', required=True,
      help="configuration file")
  parser.add_argument(
      '-t', '--task', required=True,
      help="train, eval, continue, eval_all_steps, show_best")
  parser.add_argument(
      '-r', '--restore_path', required=False,
      default="", help="ckpt path")
  parser.add_argument(
      '--model_id', required=False,
      default="", type=str, help="ckpt path")
  args = parser.parse_args()
  job = Tasks(args)
  job.run()
