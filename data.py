"""Train on simple Python functions
"""

import code
import jsonlines
import os

from tqdm import tqdm
import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.abspath(__file__))
code_data_dir = os.path.join(file_dir, 'code_data')


def read_code_dataset(subdir='train', read_all=False):
  train_data_dir = os.path.join(code_data_dir, subdir)
  data = []
  for i, train_data_file in tqdm(enumerate(os.listdir(train_data_dir))):
    data_file_path = os.path.join(train_data_dir, train_data_file)
    with jsonlines.open(data_file_path) as reader:
      for data_line in reader:
        data.append(data_line)
    if i > 0 and not read_all:
      break
  return data


def code_dataset_stats(data):
  # see the average length and try filtering out to only the shortest functions
  pass


data = code_dataset_stats()
