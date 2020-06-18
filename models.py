"""Models, optim, and loss functions based on CFG
"""

import code
from types import MethodType

import tensorflow as tf

from cfg import get_config; CFG = get_config()
from language import LangEncoder, LangDecoder, \
  sequence_reconstruction_loss, sequence_reconstruction_accuracy
from generator import Generator
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
  Generator: CFG['generator_lr'],
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
      if not any([var.name.startswith(s) for s in ['difficulty', 'current_lr']]):
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
    result = self.call(x)
    y = result['prediction']
    losses['recon_loss'] = self.loss_fn(x, y, self.loss_object) / self.global_batch_size
    for sub_model in self.layers:
      sub_model_loss = tf.math.reduce_sum(sub_model.losses)
      losses[f"{sub_model.name}_reg_loss"] = sub_model_loss

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

  return {**losses, **metrics, **result['metadata'], 'current_lr': self.current_lr}


def test_step(self, x):
  """Test step that works for all models
  """
  losses = {}
  metrics = {}
  y = self.call(x)['prediction']
  losses['val_recon_loss'] = self.loss_fn(x, y, self.loss_object) / self.global_batch_size
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
    self.current_lr = tf.Variable(
      tf.convert_to_tensor(1., tf.float32),
      trainable=False,
      name='current_lr'
    )
    print(f"AUTOENCODER LATENT DIM: {self.latent_size}")
    for r in range(CFG['lang_autoencoder_r'] + 1):
      scale = 2 ** (r)
      num_features = CFG['language_model_size'] // scale
      down = tf.keras.layers.Dense(num_features, **dense_settings)
      up = tf.keras.layers.Dense(num_features, **dense_settings)
      self.funnel_down.append(down)
      self.funnel_up.insert(0, up)


  def compile(self, optimizer, loss_fn, loss_object , metric_fn, num_replicas, debug=True):
    super(LangAutoencoder, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.loss_object = loss_object
    self.metric_fn = metric_fn
    self.train_step = MethodType(train_step, self)
    self.test_step = MethodType(test_step, self)
    self.global_batch_size = num_replicas * CFG['batch_size']


  def call(self, tokens):
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    x = self.lang_encoder(tokens)

    for down in self.funnel_down:
      x = down(x)
      x = tf.nn.leaky_relu(x)
      print(x.shape)
    
    Z = x

    for up in self.funnel_up:
      x = up(x)
      x = tf.nn.leaky_relu(x)
      print(x.shape)

    seq_pred = self.lang_decoder(x)
    return {
      'prediction': seq_pred,
      'metadata': {
        'Z': Z
      }
    }


class VisionModel(tf.keras.Model):
  def __init__(self):
    super(VisionModel, self).__init__()
    self._name = 'root'
    self.generator = Generator()
    self.decoder = BiT()
    self.symbol_embed = tf.keras.layers.Dense(CFG['vision_model_size'], **dense_settings)
    self.symbol_predict = tf.keras.layers.Dense(CFG['num_symbols'], **dense_settings)
    self.noisy_channel = get_noisy_channel()
    self.difficulty = tf.Variable(
      tf.convert_to_tensor(0, tf.int32),
      trainable=False,
      name='difficulty'
    )
    self.current_lr = tf.Variable(
      tf.convert_to_tensor(1., tf.float32),
      trainable=False,
      name='current_lr'
    )


  def compile(self, optimizer, loss_fn, loss_object , metric_fn, num_replicas, debug=True):
    super(VisionModel, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.loss_object = loss_object
    self.metric_fn = metric_fn
    self.train_step = MethodType(train_step, self)
    self.test_step = MethodType(test_step, self)
    self.global_batch_size = num_replicas * CFG['batch_size']


  def call(self, x):
    Z = self.symbol_embed(x)
    imgs = self.generator(Z)
    if CFG['use_aug']:
      aug_imgs = self.noisy_channel(imgs, self.difficulty)
    else:
      aug_imgs = imgs
    Z_pred = self.decoder(aug_imgs)
    x_pred = self.symbol_predict(Z_pred)
    return {
      'prediction': x_pred,
      'metadata': {
        'imgs': imgs,
        'aug_imgs': aug_imgs,
        'Z': Z,
        'Z_pred': Z_pred
      }
    }


class GestaltModel(tf.keras.Model):
  def __init__(self):
    super(GestaltModel, self).__init__()
    self._name = 'root'
    # LANGUAGE
    self.lang_encoder = LangEncoder()
    self.lang_decoder = LangDecoder()
    # VISION
    self.generator = Generator()
    self.decoder = BiT()
    self.noisy_channel = get_noisy_channel()
    self.difficulty = tf.Variable(
      tf.convert_to_tensor(0, tf.int32),
      trainable=False,
      name='difficulty'
    )
    self.current_lr = tf.Variable(
      tf.convert_to_tensor(1., tf.float32),
      trainable=False,
      name='current_lr'
    )
    # AUTOENCODER
    self.funnel_down = []
    self.funnel_up = []
    num_features = CFG['language_model_size']
    while num_features != CFG['target_token_dim']:
      if num_features // 4 < CFG['target_token_dim']:
        num_features = CFG['target_token_dim']
      else:
        num_features = num_features // 2
      down = tf.keras.layers.Dense(num_features, **dense_settings)
      up = tf.keras.layers.Dense(num_features * 2, **dense_settings)
      self.funnel_down.append(down)
      self.funnel_up.insert(0, up)


  def compile(self, optimizer, loss_fn, loss_object , metric_fn, num_replicas, debug=True):
    super(GestaltModel, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.loss_object = loss_object
    self.metric_fn = metric_fn
    self.train_step = MethodType(train_step, self)
    self.test_step = MethodType(test_step, self)
    self.global_batch_size = num_replicas * CFG['batch_size']


  def call(self, tokens):
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens

    # LANG
    x = self.lang_encoder(tokens)

    # DOWN
    for down in self.funnel_down:
      x = down(x)
      x = tf.nn.leaky_relu(x)
      print(x.shape)
    
    # VISION
    Z = tf.reshape(x, [CFG['batch_size'], -1])
    imgs = self.generator(Z)
    if CFG['use_aug']:
      aug_imgs = self.noisy_channel(imgs, self.difficulty)
    else:
      aug_imgs = imgs
    Z_pred = self.decoder(aug_imgs)
    # now: unfortunately some potential data loss. hey, it happens
    Z_pred = Z_pred[:, :CFG['vision_model_size']]

    # UP
    x_out = tf.reshape(Z_pred, [CFG['batch_size'], CFG['max_len'], -1])
    for up in self.funnel_up:
      x_out = up(x_out)
      x_out = tf.nn.leaky_relu(x_out)
      print(x_out.shape)

    # LANG
    seq_pred = self.lang_decoder(x_out)
    return {
      'prediction': seq_pred,
      'metadata': {
        'x': x,
        'Z': Z,
        'imgs': imgs,
        'aug_imgs': aug_imgs,
        'Z_pred': Z_pred,
        'x_out': x_out
      }
    }


# ============================== MODEL OBJECTS ==============================
def get_model():
  if CFG['full_model'] == 'language':
    model = LangAutoencoder()
  elif CFG['full_model'] == 'vision':
    return VisionModel()
  elif CFG['full_model'] == 'gestalt':
    return GestaltModel()
  else:
    raise ValueError
  return model


def crossentropy_loss():
  def loss_fn(true, pred, loss_object):
    loss = loss_object(true, pred)
    loss = tf.math.reduce_sum(loss)
    return loss
  loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=CFG['label_smoothing'],
    reduction=tf.keras.losses.Reduction.SUM
  )
  return loss_fn, loss_object


def get_loss_fn():
  if CFG['full_model'] in ['language', 'gestalt']:
    loss_fn, loss_object = sequence_reconstruction_loss()
    return loss_fn, loss_object
  elif CFG['full_model'] in ['vision']:
    loss_fn, loss_object = crossentropy_loss()
    return loss_fn, loss_object
  else:
    raise ValueError


def get_metric_fn():
  if CFG['full_model'] in ['language', 'gestalt']:
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


class ImageSnapshotManager(tf.keras.callbacks.Callback):
  def __init__(self, log_dir):
    super(ImageSnapshotManager, self).__init__()
    self.writer = tf.summary.create_file_writer(log_dir)


  def on_epoch_end(self, epoch, logs):
    if epoch % CFG['img_summaries_every_n_epochs'] == 0:
      imgs = logs['imgs']
      aug_imgs = logs['aug_imgs']
      rand_idx = tf.random.uniform([CFG['img_summaries_per_epoch']], 0, CFG['batch_size'] -1, dtype=tf.int32)
      # gather random images
      imgs = tf.gather(imgs, rand_idx)
      aug_imgs = tf.gather(aug_imgs, rand_idx)
      # concatenate together
      imgs = tf.concat(tf.unstack(imgs, axis=0), axis=1)
      aug_imgs = tf.concat(tf.unstack(aug_imgs, axis=0), axis=1)
      img = tf.concat([imgs, aug_imgs], axis=0)[tf.newaxis]
      # upload single snapshot collage
      with self.writer.as_default():
        tf.summary.image('snapshot', img, step=epoch)


class LearningRateManager(tf.keras.callbacks.Callback):
  def __init__(self):
    super(LearningRateManager, self).__init__()
    self.steps = 0

  def on_train_batch_end(self, batch, logs=None):
    self.steps += 1
    current_lr = 1
    if CFG['use_warmup'] and self.steps < CFG['warmup_steps']:
      current_lr = 1 * (self.steps / CFG['warmup_steps'])
    elif CFG['use_decay']:
      relative_steps = self.steps
      if CFG['use_warmup']:
        relative_steps = self.steps - CFG['warmup_steps']
      current_lr = 1 * CFG['decay_rate'] ** (relative_steps / CFG['decay_steps'])

    self.model.current_lr.assign(current_lr)
    self.model.optimizer.lr = current_lr


if __name__ == "__main__":
  model = get_model()
  loss_fn = get_loss_fn()
  optim = get_optim()
