import os
import sys

def print_and_exit(msg):
  print >> sys.stderr, msg
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)

def merge_two_dicts(x, y):
  '''Given two dicts, merge them into a new dict as a shallow copy.'''
  z = x.copy()
  z.update(y)
  return z
