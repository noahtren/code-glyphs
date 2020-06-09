"""Models, optim, and loss functions based on CFG
"""

import tensorflow as tf

from cfg import get_config; CFG = get_config()
from language import LangEncoder, LangDecoder, \
  sequence_reconstruction_loss, sequence_reconstruction_accuracy
from adamlrm import AdamLRM


class LangAutoencoder(tf.keras.Model):
  def __init__(self):
    super(LangAutoencoder, self).__init__()
    self.lang_encoder = LangEncoder()
    self.lang_decoder = LangDecoder()


  def compile(self, optimizer, loss_fn, metric_fn):
    super(LangAutoencoder, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.metric_fn = metric_fn


  def train_step(self, data):
    pass


  def call(tokens):
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    features = self.lang_encoder(tokens)
    seq_pred = self.lang_decoder(features)
    return seq_pred


def get_model():
  if CFG['full_model'] == 'language':
    model = LangAutoencoder()
  else:
    raise ValueError
  return model


def get_loss_fn():
  if CFG['full_model'] == 'language':
    return sequence_reconstruction_loss
  else:
    raise ValueError


def get_metric_fn():
  metric_fn = sequence_reconstruction_accuracy
  return metric_fn


def get_optim():
  lr_multiplier = {
    'root_model/lang_encoder': CFG['lang_encoder_lr'],
    'root_model/lang_decoder': CFG['lang_decoder_lr'],
    # TODO: add vision learning rates
  }
  optim = AdamLRM(lr=1., lr_multiplier=lr_multiplier)
  return optim



if __name__ == "__main__":
  model = get_model()
  loss_fn = get_loss_fn()
  optim = get_optim()
