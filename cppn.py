import code

import tensorflow as tf

from cfg import get_config; CFG = get_config()

dense_settings = {
  'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
  'bias_regularizer': tf.keras.regularizers.l2(1e-4),
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
  spatial_scale = 1. / max([y_dim, x_dim])
  spatial_scale = 1
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


class CPPN(tf.keras.Model):
  """Compositional Pattern-Producing Network
  Embeds each (x,y,r) -- where r is radius from center -- pair triple via a
  series of dense layers, combines with graph representation (z) and regresses
  pixel values directly.
  """
  def __init__(self):
    super(CPPN, self).__init__()
    self._name = 'cppn'
    self.loc_embed = tf.keras.layers.Dense(CFG['vision_hidden_size'] // 4, **dense_settings)
    self.Z_embed = tf.keras.layers.Dense(CFG['vision_hidden_size'] // 4, **dense_settings)
    self.in_w = tf.keras.layers.Dense(CFG['vision_hidden_size'], **dense_settings)
    self.ws = [tf.keras.layers.Dense(CFG['vision_hidden_size'], **dense_settings)
      for _ in range(CFG['cppn_layers'])]
    self.out_w = tf.keras.layers.Dense(CFG['c_out'], **dense_settings)
    self.img_dim = CFG['img_dim']
    self.spatial_scale = 1 / CFG['img_dim']


  def call(self, Z):
    batch_size = Z.shape[0]

    # get pixel locations and embed pixels
    loc = generate_scaled_coordinate_hints(1, self.img_dim, self.img_dim)
    loc = self.loc_embed(loc)
    loc = tf.tile(loc, [batch_size, 1, 1, 1])

    # concatenate Z to locations
    Z = self.Z_embed(Z)
    Z = tf.tile(Z[:, tf.newaxis, tf.newaxis], [1, self.img_dim, self.img_dim, 1])
    x = tf.concat([loc, Z], axis=-1)
    x = self.in_w(x)

    # encode
    for layer in self.ws:
      start_x = x
      x = layer(x)
      x = tf.nn.leaky_relu(x)

    x = self.out_w(x)
    x = tf.nn.softmax(x, axis=-1)
    x = (x * 2) - 1
    if x.shape[-1] == 1:
      # Copy grayscale along RGB axes for easy input into pre-trained, color-based models
      x = tf.tile(x, [1, 1, 1, 3])
    return x


def hex_to_rgb(hex_str):
  return [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]


def color_composite(imgs):
  assert tf.math.reduce_min(imgs) >= 0
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
