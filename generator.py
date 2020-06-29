import code
import math

import tensorflow as tf

from cfg import get_config; CFG = get_config()

dense_settings = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
}
cnn_settings = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
  'activity_regularizer': tf.keras.regularizers.l1(5e-7)
}
activity_reg = {
  'activity_regularizer': tf.keras.regularizers.l1(1e-6)
}


def get_coord_ints(y_dim, x_dim):
  """Return a 2x2 matrix where the values at each location are equal to the
  indices of that location
  """
  ys = tf.range(y_dim)[tf.newaxis]
  xs = tf.range(x_dim)[:, tf.newaxis]
  coord_ints = tf.stack([ys+xs-ys, xs+ys-xs], axis=2)
  return coord_ints


def generate_scaled_coordinate_hints(batch_size, img_dim):
  """Generally used as the input to a CPPN, but can also augment each layer
  of a ConvNet with location hints
  """
  spatial_scale = 1. / img_dim
  coord_ints = get_coord_ints(img_dim, img_dim)
  coords = tf.cast(coord_ints, tf.float32)
  coords = tf.stack([coords[:, :, 0] * spatial_scale,
                     coords[:, :, 1] * spatial_scale], axis=-1)
  coords = tf.tile(coords[tf.newaxis], [batch_size, 1, 1, 1])
  return coords


class ResidualBlock(tf.keras.layers.Layer):
  """Vision processing block based on stacked hourglass network/resdiual block,
  but using only 1x1 convolutions to work with a CPPN.
  Uses full pre-activation from Identity Mappings in Deep Residual Networks.
  ---
  Identity Mappings in Deep Residual Networks
  https://arxiv.org/pdf/1603.05027.pdf
  ---
  """
  def __init__(self, filters:int):
    super(ResidualBlock, self).__init__()
    self.first_conv = tf.keras.layers.Conv2D(
      filters // 4,
      kernel_size=1,
      strides=1,
      padding='same',
      **cnn_settings
    )
    self.second_conv = tf.keras.layers.Conv2D(
      filters // 4,
      kernel_size=3,
      strides=1,
      padding='same',
      **cnn_settings
    )
    self.third_conv = tf.keras.layers.Conv2D(
      filters,
      kernel_size=1,
      strides=1,
      padding='same',
      **cnn_settings
    )
    self.batch_norms = [tf.keras.layers.BatchNormalization() for _ in range(3)]


  def call(self, x, do_skip=True):
    start_x = x

    x = self.batch_norms[0](x)
    x = tf.nn.leaky_relu(x)
    x = self.first_conv(x)

    x = self.batch_norms[1](x)
    x = tf.nn.leaky_relu(x)
    x = self.second_conv(x)

    x = self.batch_norms[2](x)
    x = tf.nn.leaky_relu(x)
    x = self.third_conv(x)

    if do_skip:
      x = x + start_x

    return x


class BilinearAdditiveUpsampling(tf.keras.layers.Layer):
  def __init__(self, in_features:int):
    super(BilinearAdditiveUpsampling, self).__init__()
    self.upsample = tf.keras.layers.UpSampling2D(interpolation='bilinear')
    self.in_features = in_features


  def call(self, x):
    """Sum/collapse every 2 feature channels
    """
    x = self.upsample(x)
    feature_collect = tf.reshape(tf.range(self.in_features), [-1, CFG['generator_downsample_ratio']])
    features = tf.gather(x, feature_collect, axis=-1)
    x = tf.math.reduce_sum(features, axis=-1)
    return x


class CNNGenerator(tf.keras.Model):
  """Compositional Pattern-Producing Network
  Embeds each (x,y,r) -- where r is radius from center -- pair triple via a
  series of dense layers, combines with graph representation (z) and regresses
  pixel values directly.
  """
  def __init__(self):
    super(CNNGenerator, self).__init__()
    self._name = 'generator'
    self.loc_embeds = []
    self.loc_norms = []
    self.Z_embeds = []
    self.res_blocks = []
    self.up_layer = []

    filters = CFG['generator_model_size']
    for block_i in range(CFG['generator_levels']):
      loc_embeds = [tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='same',
        ) for _ in range(3)]
      loc_norms = [tf.keras.layers.BatchNormalization() for _ in range(3)]
      Z_embeds = [tf.keras.layers.Dense(filters, **dense_settings) for _ in range(6)]
      res_blocks = [ResidualBlock(filters) for _ in range(CFG['res_blocks_per_level'])]
      up_layer = BilinearAdditiveUpsampling(filters)
      filters = filters // CFG['generator_downsample_ratio']
      self.loc_embeds.append(loc_embeds)
      self.loc_norms.append(loc_norms)
      self.Z_embeds.append(Z_embeds)
      self.res_blocks.append(res_blocks)
      self.up_layer.append(up_layer)


    self.out_conv = tf.keras.layers.Conv2D(
      CFG['c_out'],
      kernel_size=1,
      strides=1,
      padding='same',
      **cnn_settings,
      **activity_reg
    )
    self.img_dim = CFG['img_dim']
    self.spatial_scale = 1 / CFG['img_dim']
    self.inp_pad = CFG['ideal_generator_model_size'] - CFG['generator_model_size']


  def call(self, Z):
    if self.inp_pad != 0:
      # deal with any needed padding when working with sequence autoencoder
      Z = tf.concat([Z, tf.zeros((CFG['batch_size'], self.inp_pad), tf.float32)], axis=-1)

    batch_size = CFG['batch_size']
    res = self.img_dim // (2 ** (CFG['generator_levels'] - 1))
    x = None

    for block_i in range(CFG['generator_levels']):
      # get pixel locations and embed pixels
      # loc = generate_scaled_coordinate_hints(1, res, res)
      # for i, (embed, norm) in enumerate(zip(self.loc_embeds[block_i], self.loc_norms[block_i])):
      #   start_loc = loc
      #   loc = embed(loc)
      #   loc = norm(loc)
      #   loc = tf.nn.swish(loc)
      #   if i != 0: loc = loc + start_loc

      # print(f"Loc: {loc.shape}")

      # embed Z vector
      block_Z = Z
      for i, embed in enumerate(self.Z_embeds[block_i]):
        start_Z = block_Z
        block_Z = embed(block_Z)
        block_Z = tf.nn.swish(block_Z)
        if i != 0: block_Z = block_Z + start_Z

      print(f"Z: {block_Z.shape}")

      # concatenate Z to locations
      # loc = tf.tile(loc, [batch_size, 1, 1, 1])
      block_Z = tf.tile(block_Z[:, tf.newaxis, tf.newaxis], [1, res, res, 1])
      # x = loc + block_Z if x is None else loc + block_Z + x
      x = block_Z if x is None else block_Z + x
      print(f"x: {x.shape}")

      # residual layer processing
      for i, resblock in enumerate(self.res_blocks[block_i]):
        x = resblock(x)
        print(f"[Level {block_i}/ Res block {i}]: {x.shape}")
      
      x = self.up_layer[block_i](x)
      res = res * 2
      
    x = self.out_conv(x)
    x = tf.nn.sigmoid(x)
    # x = tf.nn.softmax(x, axis=-1)
    # x = color_composite(x)
    return x


class CPPNBlock(tf.keras.Model):
  """Residual block pattern adapted for a CPPN
  """
  def __init__(self, filters):
    super(CPPNBlock, self).__init__()
    self.layer_1 = tf.keras.layers.Dense(filters // 4, **dense_settings)
    self.layer_2 = tf.keras.layers.Dense(filters // 4, **dense_settings)
    self.layer_3 = tf.keras.layers.Dense(filters, **dense_settings)


  def call(self, x, do_skip=True):
    start_x = x

    x = tf.nn.leaky_relu(x)
    x = self.layer_1(x)

    x = tf.nn.leaky_relu(x)
    x = self.layer_2(x)

    x = tf.nn.leaky_relu(x)
    x = self.layer_3(x)

    if do_skip:
      start_x = tf.nn.dropout(x, 0.1)
      x = x + start_x
    return x


def initialize_fourier_mapping_vector(m, sigma):
  d = 2
  B = tf.random.normal((m, d)) * sigma
  return B


def fourier_mapping(coords, B):
  """Preprocess each coordinate — scaled from [0, 1) — by converting each
  coordinate to a random fourier feature, as determined by a matrix with values
  samples from a Gaussian distribution.
  """
  sin_features = tf.math.sin((2 * math.pi) * (tf.matmul(coords, B, transpose_b=True)))
  cos_features = tf.math.cos((2 * math.pi) * (tf.matmul(coords, B, transpose_b=True)))
  features = tf.concat([sin_features, cos_features], axis=-1)
  return features


class CPPNGenerator(tf.keras.Model):
  def __init__(self):
    super(CPPNGenerator, self).__init__()
    self.Z_embeds = [tf.keras.layers.Dense(CFG['generator_model_size'], **dense_settings) for _ in range(6)]
    BlockObject = ResidualBlock if CFG['cppn_conv'] else CPPNBlock
    self.blocks = [CPPNBlock(CFG['generator_model_size']) for _ in range(CFG['cppn_blocks'])]
    self.mlp_out = tf.keras.layers.Dense(3, **activity_reg, **dense_settings)
    self.B1 = initialize_fourier_mapping_vector(m=128, sigma=5)
    self.B2 = initialize_fourier_mapping_vector(m=128, sigma=1)


  def call(self, Z):
    batch_size = Z.shape[0]
    
    for i, embed in enumerate(self.Z_embeds):
      start_Z = Z
      Z = embed(Z)
      Z = tf.nn.swish(Z)
      if i != 0: Z = Z + start_Z
    Z = tf.tile(Z[:, tf.newaxis, tf.newaxis], [1, CFG['img_dim'], CFG['img_dim'], 1])

    coords = generate_scaled_coordinate_hints(batch_size, CFG['img_dim'])
    features1 = fourier_mapping(coords, B=self.B1)
    features2 = fourier_mapping(coords, B=self.B2)
    features = tf.concat([features1, features2], axis=-1)

    for i, block in enumerate(self.blocks):
      if i == 0:
        x = block(features, do_skip=False) + Z
      else:
        x = block(x)
      print(f"CPPN Block {i}: {x.shape}")
      x = tf.nn.swish(x)
    x = self.mlp_out(x)
    x = tf.nn.sigmoid(x)
    print(f"Img out: {x.shape}")
    return x


def hex_to_rgb(hex_str):
  return [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]


def color_composite(imgs):
  out_channels = tf.zeros(imgs.shape[:3] + [3])
  for channel_i in range(imgs.shape[3]):
    hex_str = CFG['composite_colors'][channel_i]
    rgb = tf.convert_to_tensor(hex_to_rgb(hex_str))
    rgb = tf.cast(rgb, tf.float32) / 255.
    composite_channel = imgs[..., channel_i, tf.newaxis] * rgb
    out_channels += composite_channel
  return out_channels


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  model = CPPN()
  z = tf.random.normal((1, 512))
  imgs = model(z)
  imgs = (imgs + 1) / 2.
  imgs = color_composite(imgs)
  plt.imshow(imgs[0]); plt.show()
