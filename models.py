"""Models, optim, and loss functions based on CFG
"""

import code
from types import MethodType

import tensorflow as tf

from cfg import get_config; CFG = get_config()
from language import LangEncoder, LangDecoder, \
  sequence_reconstruction_loss, sequence_reconstruction_accuracy
from cppn import CPPN
from bit_model import BiT
from adamlrm import AdamLRM
from aug import get_noisy_channel

dense_settings = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
}

lr_multiplier_models = {
  LangEncoder: CFG['lang_encoder_lr'],
  LangDecoder: CFG['lang_decoder_lr'],
  BiT: CFG['decoder_lr'],
  CPPN: CFG['generator_lr'],
  tf.keras.layers.Dense: CFG['default_lr']
}


# ============================== MODELING ==============================
def variable_lr_map(model):
  """Map each variable name to learning rate based on submodel
  """
  unassigned_variables = []
  unassigned_models = []
  name_lr_map = {}
  # sub model variables
  for sub_model in model.layers:
    if type(sub_model) in lr_multiplier_models:
      for var in sub_model.variables:
        name_lr_map[var.name] = lr_multiplier_models[type(sub_model)]
    else:
      unassigned_models.append(sub_model)
  for var in model.variables:
    if var.name not in name_lr_map:
      if not var.name.startswith('difficulty'):
        unassigned_variables.append(var)
  if unassigned_variables != [] or unassigned_models != []:
    print("Problem: there are some unassigned values in the learning rate multiplier.")
    code.interact(local={**locals(), **globals()})
  return name_lr_map
  

def train_step(self, x):
  """Train step that works for all models
  """
  losses = {}
  metrics = {}
  with tf.GradientTape() as tape:
    y = self.call(x)['prediction']
    losses['recon_loss'] = self.loss_fn(x, y)
    for sub_model in self.layers:
      sub_model_loss = tf.math.reduce_sum(sub_model.losses)
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

  if hasattr(self, 'difficulty'): metrics['difficulty'] = self.difficulty

  return {**losses, **metrics}


def test_step(self, x):
  """Test step that works for all models
  """
  losses = {}
  metrics = {}
  y = self.call(x)['prediction']
  losses['val_recon_loss'] = self.loss_fn(x, y)
  metrics['val_recon_acc'] = self.metric_fn(x, y)
  return {**losses, **metrics}


class LangAutoencoder(tf.keras.Model):
  def __init__(self, debug=True):
    super(LangAutoencoder, self).__init__()
    self._name = 'root'
    self.lang_encoder = LangEncoder()
    self.lang_decoder = LangDecoder()
    self.latent_size = (CFG['language_model_size'] // (2 ** CFG['lang_autoencoder_r'])) * CFG['max_len']
    self.funnel_down = []
    self.funnel_up = []
    print(f"AUTOENCODER LATENT DIM: {self.latent_size}")
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
    self.train_step = MethodType(train_step, self)
    self.test_step = MethodType(test_step, self)


  def call(self, tokens):
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    x = self.lang_encoder(tokens)

    for down in self.funnel_down:
      x = down(x)
      x = tf.nn.swish(x)
      print(x.shape)
    
    Z = x

    for up in self.funnel_up:
      x = up(x)
      x = tf.nn.swish(x)
      print(x.shape)

    seq_pred = self.lang_decoder(x)
    return {
      'prediction': seq_pred,
      'Z': Z
    }


class VisionModel(tf.keras.Model):
  def __init__(self):
    super(VisionModel, self).__init__()
    self._name = 'root'
    self.generator = CPPN()
    self.decoder = BiT()
    self.symbol_embed = tf.keras.layers.Dense(CFG['vision_hidden_size'], **dense_settings)
    self.symbol_predict = tf.keras.layers.Dense(CFG['num_symbols'], **dense_settings)
    self.noisy_channel = get_noisy_channel()
    self.difficulty = tf.Variable(
      tf.convert_to_tensor(0, tf.int32),
      trainable=False,
      name='difficulty'
    )


  def compile(self, optimizer, loss_fn, metric_fn, debug=True):
    super(VisionModel, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.metric_fn = metric_fn
    self.train_step = MethodType(train_step, self)
    self.test_step = MethodType(test_step, self)


  def call(self, x):
    Z = self.symbol_embed(x)
    imgs = self.generator(Z)
    if CFG['use_aug']:
      aug_imgs = self.noisy_channel(imgs, self.difficulty)
    else:
      aug_imgs = None
    Z_pred = self.decoder(imgs)
    x_pred = self.symbol_predict(Z_pred)
    return {
      'prediction': x_pred,
      'imgs': imgs,
      'aug_imgs': aug_imgs
    }


# ============================== MODEL OBJECTS ==============================
def get_model():
  if CFG['full_model'] == 'language':
    model = LangAutoencoder()
  elif CFG['full_model'] == 'vision':
    return VisionModel()
  else:
    raise ValueError
  return model


def get_loss_fn():
  if CFG['full_model'] in ['language']:
    return sequence_reconstruction_loss
  elif CFG['full_model'] in ['vision']:
    return tf.keras.losses.CategoricalCrossentropy(
      from_logits=True,
      label_smoothing=CFG['label_smoothing'])
  else:
    raise ValueError


def get_metric_fn():
  if CFG['full_model'] in ['language']:
    return sequence_reconstruction_accuracy
  elif CFG['full_model'] in ['vision']:
    return tf.keras.metrics.CategoricalAccuracy()
  else:
    raise ValueError



def get_optim(model):
  name_lr_map = variable_lr_map(model)
  optim = AdamLRM(lr=1., lr_multiplier=name_lr_map)
  return optim


class DifficultyManager(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if logs['recon_acc'] < 0.3 and self.model.difficulty > 0:
      self.model.difficulty.assign_add(-1)
    if logs['recon_acc'] > 0.95 and self.model.difficulty < 15:
      self.model.difficulty.assign_add(1)


if __name__ == "__main__":
  model = get_model()
  loss_fn = get_loss_fn()
  optim = get_optim()
