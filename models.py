"""Models, optim, and loss functions based on CFG
"""

import code

import tensorflow as tf

from cfg import get_config; CFG = get_config()
from language import LangEncoder, LangDecoder, \
  sequence_reconstruction_loss, sequence_reconstruction_accuracy
from adamlrm import AdamLRM

dense_settings = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
}


def print_model_prefixes(model):
  """Find the model prefix for each model
  """
  # for sub_model in model.layers:
  #   for variable in sub_model.variables:
  #     print(f"{sub_model}: {variable.name}")
  for variable in model.variables:
    print(variable.name)


class LangAutoencoder(tf.keras.Model):
  def __init__(self, debug=True):
    super(LangAutoencoder, self).__init__()
    self._name = 'lang_autoencoder'
    self.lang_encoder = LangEncoder()
    self.lang_decoder = LangDecoder()
    self.latent_size = (CFG['language_model_size'] // (2 ** CFG['lang_autoencoder_r'])) * CFG['max_len']
    self.funnel_down = []
    self.funnel_up = []
    if debug: tf.print(f"AUTOENCODER LATENT DIM: {self.latent_size}")
    for r in range(CFG['lang_autoencoder_r'] + 1):
      scale = 2 ** (r)
      num_features = CFG['language_model_size'] // scale
      down = tf.keras.layers.Dense(num_features, **dense_settings)
      up = tf.keras.layers.Dense(num_features, **dense_settings)
      self.funnel_down.append(down)
      self.funnel_up.insert(0, up)


  def compile(self, optimizer, loss_fn, metric_fn, debug=True):
    super(LangAutoencoder, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.metric_fn = metric_fn
    if debug: print_model_prefixes(self)

  # write a test step as well
  def train_step(self, data):
    x = data
    losses = {}
    metrics = {}
    with tf.GradientTape() as tape:
      y = self.call(x)
      losses['recon_loss'] = self.loss_fn(x, y)
      for sub_model in self.layers:
        losses[f"{sub_model.name}_reg_loss"] = tf.math.reduce_sum(sub_model.losses)

      # sum all losses
      loss_sum = 0.
      for loss in losses.values():
        loss_sum += loss

    # optimize
    grads = tape.gradient(loss_sum, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    # metrics
    metrics['recon_acc'] = self.metric_fn(x, y)

    return {**losses, **metrics}


  def call(self, tokens):
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    x = self.lang_encoder(tokens)

    for down in self.funnel_down:
      x = down(x)
      x = tf.nn.swish(x)
      print(x.shape)

    for up in self.funnel_up:
      x = up(x)
      x = tf.nn.swish(x)
      print(x.shape)

    # x = tf.reshape(x, [CFG['batch_size'], CFG['max_len'], -1])
    seq_pred = self.lang_decoder(x)
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
    'tf_roberta_model': CFG['lang_encoder_lr'],
    'tf_roberta_for_masked_lm': CFG['lang_decoder_lr'],
    'lang_autoencoder': 0.001 # default adam learning rate
    # TODO: add vision learning rates
  }
  optim = AdamLRM(lr=1., lr_multiplier=lr_multiplier)
  return optim



if __name__ == "__main__":
  model = get_model()
  loss_fn = get_loss_fn()
  optim = get_optim()
