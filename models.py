"""Models, optim, and loss functions based on CFG
"""

import code
from types import MethodType

import tensorflow as tf

from cfg import get_config; CFG = get_config()
from language import LangEncoder, LangDecoder, \
  sequence_reconstruction_loss, sequence_reconstruction_accuracy
from generator import CNNGenerator, CPPNGenerator
from bit_model import BiT
from adamlrm import AdamLRM
from aug import get_noisy_channel

dense_settings = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
}


class FunnelDown(tf.keras.Model):
  """Reduce representations of each token.
  """
  def __init__(self):
    super(FunnelDown, self).__init__()
    self.l = []
    num_features = CFG['language_model_size']
    while num_features != CFG['target_down_token_dim']:
      if num_features // 4 < CFG['target_down_token_dim']:
        num_features = CFG['target_down_token_dim']
      else:
        num_features = num_features // 2
      down = tf.keras.layers.Dense(num_features, **dense_settings)
      self.l.append(down)
    self.l = tuple(self.l)


  def call(self, x):
    for layer in self.l:
      x = layer(x)
      x = tf.nn.leaky_relu(x)
      print(x.shape)
    return x


class FunnelUp(tf.keras.Model):
  """Upsample representations of each token
  """
  def __init__(self):
    super(FunnelUp, self).__init__()
    self.l = []
    num_features = CFG['language_model_size']
    while num_features != CFG['target_up_token_dim']:
      if num_features // 4 < CFG['target_up_token_dim']:
        num_features = CFG['target_up_token_dim']
      else:
        num_features = num_features // 2
      up = tf.keras.layers.Dense(num_features * 2, **dense_settings)
      self.l.insert(0, up)
    self.l = tuple(self.l)


  def call(self, x):
    for layer in self.l:
      x = layer(x)
      x = tf.nn.leaky_relu(x)
      print(x.shape)
    return x


lr_multiplier_models = {
  LangEncoder: CFG['lang_encoder_lr'],
  LangDecoder: CFG['lang_decoder_lr'],
  BiT: CFG['decoder_lr'],
  CNNGenerator: CFG['generator_lr'],
  CPPNGenerator: CFG['generator_lr'],
  FunnelDown: CFG['funnel_lr'],
  FunnelUp: CFG['funnel_lr'],
  tf.keras.layers.Dense: CFG['default_lr']
}


# ============================== HELPER FUNCTIONS ==============================
def perceptual_loss(features, max_pairs=1_000, MULTIPLIER=1):
  """Return a negative value, where greater magnitudes describe further distances.
  This is to encourage samples to be perceptually more distant from each other.
  i.e., they are repelled from each other.
  """
  b_s = features.shape[0]
  # batch size must be at least 2 for this to be effective at all
  assert b_s >= 2

  # adapted from https://github.com/tensorflow/lucid/blob/master/lucid/optvis/objectives.py
  flattened = tf.reshape(features, [b_s, -1, features.shape[-1]])
  grams = tf.matmul(flattened, flattened, transpose_a=True)
  grams = tf.nn.l2_normalize(flattened, axis=[1,2], epsilon=1e-10)

  num_pairs = (b_s * (b_s - 1)) // 2
  pair_idxs = tf.where(tf.ones((b_s, b_s)))
  non_matching_pairs = tf.squeeze(tf.where(pair_idxs[:, 0] < pair_idxs[:, 1]))
  pair_idxs = tf.gather(pair_idxs, non_matching_pairs)

  num_pairs = min([num_pairs, max_pairs])
  use_pair_idxs = tf.random.uniform([num_pairs], minval=0, maxval=num_pairs, dtype=tf.int32)
  pair_idxs = tf.gather(pair_idxs, use_pair_idxs)

  gram_pairs = tf.gather(grams, pair_idxs)
  diffs = tf.math.multiply(gram_pairs[:, 0], gram_pairs[:, 1])
  percept_loss = tf.math.reduce_sum(diffs) / num_pairs
  percept_loss = percept_loss * MULTIPLIER
  return percept_loss


def vector_distance_loss(rep1, rep2, max_pairs=1_000):
  """Experimental loss score based on trying to match the distance between
  corresponding pairs of representations, using L2 loss between distances
  normalized from 0 to 1. This is more stable than using logistic loss.
  This use case: match the distances between pairs of images perceptions with
  the distances between pairs of symbol labels.
  Update: this is likely not necessary because graph pretraining actually hasn't
  shown to provide a major qualitative improvement, so there is nothing
  particularly special about the graph embeddings vs. visual latent spaces.
  Still curious!
  """
  n = rep1.shape[0]
  # batch size must be at least 2 for this to be effective at all
  assert n >= 2

  num_pairs = (rep1.shape[0] * (rep1.shape[0] - 1)) // 2
  pairs = tf.where(tf.ones((n, n)))

  # get diagonal of adjacency matrix indices
  unique_pairs = tf.squeeze(tf.where(pairs[:, 0] < pairs[:, 1]))
  pairs = tf.gather(pairs, unique_pairs)
  num_pairs = min([num_pairs, max_pairs])
  use_pair_idxs = tf.random.uniform([num_pairs], minval=0, maxval=num_pairs, dtype=tf.int32)
  pairs = tf.gather(pairs, use_pair_idxs)
  
  # DISTANCES 1
  pairs1 = tf.gather(rep1, pairs)
  diffs1 = tf.abs(pairs1[:, 0] - pairs1[:, 1])
  # get average feature distance
  diffs1 = tf.math.reduce_mean(diffs1, axis=tf.range(tf.rank(diffs1) - 1) + 1)

  # DISTANCES 2
  pairs2 = tf.gather(rep2, pairs)
  diffs2 = tf.abs(pairs2[:, 0] - pairs2[:, 1])
  # get average feature distance
  diffs2 = tf.math.reduce_mean(diffs2, axis=tf.range(tf.rank(diffs2) - 1) + 1)

  # NORMALIZE DISTANCES
  def zero_one_normalize(tensor, epsilon=1e-7):
    tensor = tensor - tf.math.reduce_min(tensor)
    tensor = tensor + epsilon
    tensor = tensor / tf.math.reduce_max(tensor)
    return tensor

  diffs1 = zero_one_normalize(diffs1)
  diffs2 = zero_one_normalize(diffs2)
  
  # FIND THE DISTANCE BETWEEN DISTANCES (haha)
  diffs1 = diffs1[:, tf.newaxis]
  diffs2 = diffs2[:, tf.newaxis]
  error = tf.keras.losses.mean_squared_error(diffs1, diffs2)
  # error = tf.keras.losses.binary_crossentropy(diffs1, diffs2)
  error = tf.math.reduce_mean(error)
  # this is positive because we want points to be similarly distance
  # regardless of the representation medium
  return error


# ============================== GET MODELS ==============================
def get_generator():
  if CFG['generator_model'] == 'cnn':
    return CNNGenerator()
  elif CFG['generator_model'] == 'cppn':
    return CPPNGenerator()
  else:
    raise RuntimeError()


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

    # perceptual loss
    if CFG['use_perceptual_loss']:
      percept_loss = perceptual_loss(
        result['metadata']['percept'],
        # result['metadata']['Z']
      )
      losses['percept_loss'] = percept_loss

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
      tf.convert_to_tensor(0., tf.float32),
      trainable=False,
      name='current_lr'
    )
    self.funnel_down = FunnelDown()
    self.funnel_up = FunnelUp()


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

    x = self.funnel_down(x)
    
    Z = x

    x = self.funnel_up(x)

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
    self.generator = get_generator()
    self.decoder = BiT()
    self.perceptor = None
    if CFG['use_perceptual_loss']:
      self.perceptor = BiT(percept=True)
      self.perceptor.trainable = False
    self.symbol_embed = tf.keras.layers.Dense(CFG['vision_model_size'], **dense_settings)
    self.symbol_predict = tf.keras.layers.Dense(CFG['num_symbols'], **dense_settings)
    self.noisy_channel = get_noisy_channel()
    self.difficulty = tf.Variable(
      tf.convert_to_tensor(0, tf.int32),
      trainable=False,
      name='difficulty'
    )
    self.current_lr = tf.Variable(
      tf.convert_to_tensor(0., tf.float32),
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
    print(f"Z pred: {Z_pred.shape}")
    if CFG['use_perceptual_loss']:
      _, percept = self.perceptor(aug_imgs, perceptual=True)
      print(f"percept: {percept.shape}")
    else:
      percept = 0.
    x_pred = self.symbol_predict(Z_pred)
    return {
      'prediction': x_pred,
      'metadata': {
        'imgs': imgs,
        'aug_imgs': aug_imgs,
        'percept': percept,
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
    self.generator = get_generator()
    self.decoder = BiT()
    self.perceptor = None
    if CFG['use_perceptual_loss']:
      self.perceptor = BiT(percept=True)
      self.perceptor.trainable = False
    self.noisy_channel = get_noisy_channel()
    self.difficulty = tf.Variable(
      tf.convert_to_tensor(0, tf.int32),
      trainable=False,
      name='difficulty'
    )
    self.current_lr = tf.Variable(
      tf.convert_to_tensor(0., tf.float32),
      trainable=False,
      name='current_lr'
    )
    self.funnel_down = FunnelDown()
    self.funnel_up = FunnelUp()


  def compile(self, optimizer, loss_fn, loss_object , metric_fn, num_replicas, debug=True):
    super(GestaltModel, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.loss_object = loss_object
    self.metric_fn = metric_fn
    self.train_step = MethodType(train_step, self)
    self.test_step = MethodType(test_step, self)
    self.global_batch_size = num_replicas * CFG['batch_size']


  def custom_save(self, path):
    self.generator.save(
      f"{path}/generator/best",
      include_optimizer=False,
      save_format='tf',
    )
    print("Saved generator")
    self.decoder.save(
      f"{path}/decoder/best",
      include_optimizer=False,
      save_format='tf',
    )
    print("Saved decoder")
    self.funnel_up.save(
      f"{path}/funnel-up/best",
      include_optimizer=False,
      save_format='tf',
    )
    self.funnel_down.save(
      f"{path}/funnel-down/best",
      include_optimizer=False,
      save_format='tf',
    )
    print("Saved funnel up and funnel down")
    self.lang_encoder.save(
      f"{path}/lang-encoder/best",
      include_optimizer=False,
      save_format='tf',
    )
    self.lang_decoder.save(
      f"{path}/lang-decoder/best",
      include_optimizer=False,
      save_format='tf',
    )
    print("Saved lang encoder and decoder")


  def call(self, tokens):
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens

    # LANG
    x = self.lang_encoder(tokens)

    x = self.funnel_down(x)
    
    # VISION
    Z = tf.reshape(x, [CFG['batch_size'], -1])
    imgs = self.generator(Z)
    if CFG['use_aug']:
      aug_imgs = self.noisy_channel(imgs, self.difficulty)
    else:
      aug_imgs = imgs
    Z_pred = self.decoder(aug_imgs)
    # now: unfortunately some potential data loss. hey, it happens
    if Z_pred.shape[1] < CFG['vision_model_size']:
      Z_pred = Z_pred[:, :CFG['vision_model_size']]
    print(f"Z pred: {Z_pred.shape}")
    if CFG['use_perceptual_loss']:
      _, percept = self.perceptor(aug_imgs, perceptual=True)
      print(f"percept: {percept.shape}")
    else:
      percept = 0.

    # UP
    x_out = tf.reshape(Z_pred, [CFG['batch_size'], CFG['max_len'], -1])
    print(x_out.shape)
    
    x_out = self.funnel_up(x_out)

    # LANG
    seq_pred = self.lang_decoder(x_out)
    return {
      'prediction': seq_pred,
      'metadata': {
        'x': x,
        'Z': Z,
        'imgs': imgs,
        'aug_imgs': aug_imgs,
        'percept': percept,
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
  optim = AdamLRM(lr=0., lr_multiplier=name_lr_map)
  return optim


# ============================== CALLBACKS ==============================


def get_best_acc(difficulty):
  """Linear increase in required accuracy before going to the
  next difficulty level. Part of curriculum learning.
  """
  return 0.9 + 0.1 * (difficulty / 16)


class DifficultyManager(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if logs['recon_acc'] < 0.3 and self.model.difficulty > 0:
      self.model.difficulty.assign_add(-1)
    target_acc = get_best_acc(self.model.difficulty)
    if logs['recon_acc'] > target_acc and self.model.difficulty < 15:
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


class CheckpointSaver(tf.keras.callbacks.Callback):
  def __init__(self):
    self.best_acc = 0.
    self.ckpt_counter = CFG['ckpt_every_n_epochs']
  
  def on_epoch_end(self, epoch, logs):
    if self.ckpt_counter > 0:
      self.ckpt_counter -= 1
    else:
      capped_acc = min([logs['recon_acc'], get_best_acc(self.model.difficulty)])
      if capped_acc > self.best_acc:
        self.best_acc = capped_acc
        print(f"\nSaving new weights with best accuracy of {capped_acc}")
        self.model.custom_save(
          f"{CFG['path_prefix']}checkpoints/{CFG['run_name']}",
        )
        self.ckpt_counter = CFG['ckpt_every_n_epochs']


if __name__ == "__main__":
  model = get_model()
  loss_fn = get_loss_fn()
  optim = get_optim()
