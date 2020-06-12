"""Train on simple Python functions
"""

import code
import os
import json
import re

import jsonlines
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

from cfg import get_config; CFG = get_config()
from language import batch_encode, tokenizer

file_dir = os.path.dirname(os.path.abspath(__file__))
code_data_dir = os.path.join(file_dir, 'code_data')


def read_code_dataset(subdir='train', read_all=True):
  train_data_dir = os.path.join(code_data_dir, subdir)
  pattern = r'(\"\"\"(.|\n)*\"\"\"\s*)'
  data = []
  for i, train_data_file in tqdm(enumerate(os.listdir(train_data_dir))):
    if i > 0 and not read_all:
      break
    data_file_path = os.path.join(train_data_dir, train_data_file)
    with jsonlines.open(data_file_path) as reader:
      for data_line in reader:
        if len(data_line['code_tokens']) <= CFG['max_len']:
          code_str = data_line['original_string']
          docstring = re.search(pattern, code_str)
          if docstring is not None:
            docstring = docstring.groups()[0]
            print(docstring)
            # if len(docstring) != 1:
            #   code.interact(local={**locals(), **globals()})
            code_str = code_str.replace(docstring, "")
          if len(tokenizer.encode(code_str)) <= CFG['max_len']:
            data.append(code_str)
  with open(f"data_cache_{CFG['max_len']}.json", "w+") as f:
    f.write(json.dumps(data))
  print(f"Got {len(data)} data samples")
  return data


def code_dataset_stats(data):
  # see the average length and try filtering out to only the shortest functions
  token_lens = []
  for func in data:
    token_lens.append(len(func['code_tokens']))
  plt.hist(token_lens, bins=300)
  

def read_small_code_dataset_cache():
  cache_loc = os.path.join(
    code_data_dir,
    f"data_cache_{CFG['max_len']}.json"
  )
  print(f"Reading dataset from {cache_loc}")
  cache_data = json.load(open(cache_loc, 'r'))
  return cache_data


def get_dataset():
  """Returns a tf.data.Dataset that produces batches of
  `input_ids` and `attention_mask` tensors.
  """
  if CFG['full_model'] in ['language']:
    data = read_small_code_dataset_cache()
    data_split = int(len(data) * 0.8)
    train_data = data[:data_split]
    test_data = data[data_split:]
    def get_code_dataset(code_lines):
      ds = tf.data.Dataset.from_tensor_slices(data)
      py_func_batch_encode = lambda strings: tf.py_function(
        func=batch_encode,
        inp=[strings],
        Tout=[tf.int32, tf.int32])
      nest_data = lambda inp_ids, atn_mask: {
        'input_ids': tf.squeeze(inp_ids),
        'attention_mask': tf.squeeze(atn_mask)
      }
      def set_shapes(x):
        for key in x.keys():
          x[key].set_shape((CFG['max_len']))
        return x
      ds = ds.map(py_func_batch_encode)
      ds = ds.map(nest_data)
      ds = ds.map(set_shapes)
      ds = ds.batch(CFG['batch_size'])
      return ds
    ds = get_code_dataset(train_data)
    val_ds = get_code_dataset(test_data)
    return ds, val_ds
  else:
    raise NotImplementedError


if __name__ == "__main__":
  # read_code_dataset()

  ds, val_ds = get_dataset()
  for val in ds.take(1): pass
