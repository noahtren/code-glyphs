import code

import tensorflow as tf

from cfg import get_config; CFG = get_config()

dense_settings = {
  # 'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  # 'bias_regularizer': tf.keras.regularizers.l2(1e-4),
}
cnn_settings = {

}
activity_reg = {
  'activity_regularizer': tf.keras.regularizers.l1(5e-5)
}


def get_coord_ints(y_dim, x_dim):
  ys = tf.range(y_dim)[tf.newaxis]
  xs = tf.range(x_dim)[:, tf.newaxis]
  coord_ints = tf.stack([ys+xs-ys, xs+ys-xs], axis=2)
  return coord_ints


def generate_scaled_coordinate_hints(batch_size, y_dim, x_dim):
  """Generally used as the input to a CPPN, but can also augment each layer
  of a ConvNet with location hints
  """
  # spatial_scale = 1. / max([y_dim, x_dim])
  spatial_scale = 1.
  coord_ints = get_coord_ints(y_dim, x_dim)
  coords = tf.cast(coord_ints, tf.float32)
  coords = tf.stack([coords[:, :, 0] * spatial_scale,
                      coords[:, :, 1] * spatial_scale], axis=-1)
  h = x_dim // 2
  dists = tf.stack([coords[:, :, 0] - h,
                    coords[:, :, 1] - h], axis=-1)
  r = tf.sqrt(tf.math.reduce_sum(dists ** 2, axis=-1))[..., tf.newaxis]
  loc = tf.concat([dists, r], axis=-1)
  loc = tf.tile(loc[tf.newaxis], [batch_size, 1, 1, 1])
  return loc


class Conv1x1ResidualBlock(tf.keras.layers.Layer):
  """Vision processing block based on stacked hourglass network/resdiual block,
  but using only 1x1 convolutions to work with a CPPN.
  Uses full pre-activation from Identity Mappings in Deep Residual Networks.
  ---
  Identity Mappings in Deep Residual Networks
  https://arxiv.org/pdf/1603.05027.pdf
  ---
  """
  def __init__(self, filters:int):
    super(Conv1x1ResidualBlock, self).__init__()
    self.first_conv = tf.keras.layers.Conv2D(
      filters // 4,
      kernel_size=1,
      strides=1,
      padding='same',
      **cnn_settings
    )
    self.second_conv = tf.keras.layers.Conv2D(
      filters // 4,
      kernel_size=1,
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


  def call(self, x):
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

    x = x + start_x
    return x


class CPPN(tf.keras.Model):
  """Compositional Pattern-Producing Network
  Embeds each (x,y,r) -- where r is radius from center -- pair triple via a
  series of dense layers, combines with graph representation (z) and regresses
  pixel values directly.
  """
  def __init__(self):
    super(CPPN, self).__init__()
    self._name = 'cppn'
    self.loc_embeds = [tf.keras.layers.Conv2D(
      CFG['vision_hidden_size'] // 4,
      kernel_size=1,
      strides=1,
      padding='same',
      **cnn_settings
      ) for _ in range(3)]
    self.loc_norms = [tf.keras.layers.BatchNormalization() for _ in range(3)]

    self.Z_embeds = [tf.keras.layers.Dense(CFG['vision_hidden_size'], **dense_settings) for _ in range(3)]
    self.Z_norms = [tf.keras.layers.BatchNormalization() for _ in range(3)]

    self.in_conv = tf.keras.layers.Conv2D(
      CFG['vision_hidden_size'],
      kernel_size=1,
      strides=1,
      padding='same',
      **cnn_settings
    )
    self.res_blocks = [Conv1x1ResidualBlock(CFG['vision_hidden_size']) for _ in range(CFG['cppn_layers'])]
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


  def call(self, Z):
    batch_size = CFG['batch_size']

    # get pixel locations and embed pixels
    loc = generate_scaled_coordinate_hints(1, self.img_dim, self.img_dim)
    for i, (embed, norm) in enumerate(zip(self.loc_embeds, self.loc_norms)):
      start_loc = loc
      loc = embed(loc)
      loc = norm(loc)
      loc = tf.nn.leaky_relu(loc)
      if i != 0: loc = loc + start_loc

    # concatenate Z to locations
    for i, (embed, norm) in enumerate(zip(self.Z_embeds, self.Z_norms)):
      start_Z = Z
      Z = embed(Z)
      Z = norm(Z)
      Z = tf.nn.leaky_relu(Z)
      if i != 0: Z = Z + start_Z

    loc = tf.tile(loc, [batch_size, 1, 1, 1])
    Z = tf.tile(Z[:, tf.newaxis, tf.newaxis], [1, self.img_dim, self.img_dim, 1])
    x = tf.concat([loc, Z], axis=-1)
    x = self.in_conv(x)

    # generate
    for resblock in self.res_blocks:
      x = resblock(x)
    
    x = self.out_conv(x)
    x = tf.nn.softmax(x, axis=-1)
    x = color_composite(x)
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
