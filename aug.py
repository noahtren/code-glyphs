import code
import random
import imageio
import math

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from cfg import get_config; CFG = get_config()


def gaussian_k(height, width, y, x, sigma, normalized=True):
  """Make a square gaussian kernel centered at (x, y) with sigma as standard deviation.
  Returns:
      A 2D array of size [height, width] with a Gaussian kernel centered at (x, y)
  """
  # cast arguments used in calculations
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)
  sigma = tf.cast(sigma, tf.float32)
  # create indices
  xs = tf.range(0, width, delta=1., dtype=tf.float32)
  ys = tf.range(0, height, delta=1., dtype=tf.float32)
  ys = tf.expand_dims(ys, 1)
  # apply gaussian function to indices based on distance from x, y
  gaussian = tf.math.exp(-((xs - x)**2 + (ys - y)**2) / (2 * (sigma**2)))
  if normalized:
      gaussian = gaussian / tf.math.reduce_sum(gaussian) # all values will sum to 1
  return gaussian


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
  """Return a 3x3 transformmatrix which transforms indicies of original images
  """

  # CONVERT DEGREES TO RADIANS
  rotation = math.pi * rotation / 180.
  shear = math.pi * shear / 180.

  # ROTATION MATRIX
  c1 = tf.math.cos(rotation)
  s1 = tf.math.sin(rotation)
  rotation_matrix = tf.reshape(tf.concat([c1,s1,[0], -s1,c1,[0], [0],[0],[1]],axis=0),[3,3])

  # SHEAR MATRIX
  c2 = tf.math.cos(shear)
  s2 = tf.math.sin(shear)
  shear_matrix = tf.reshape(tf.concat([[1],s2,[0], [0],c2,[0], [0],[0],[1]],axis=0),[3,3])    
  
  # ZOOM MATRIX
  zoom_matrix = tf.reshape( tf.concat([[1]/height_zoom,[0],[0], [0],[1]/width_zoom,[0], [0],[0],[1]],axis=0),[3,3])
  
  # SHIFT MATRIX
  shift_matrix = tf.reshape( tf.concat([[1],[0],height_shift, [0],[1],width_shift, [0],[0],[1]],axis=0),[3,3])

  return tf.matmul(tf.matmul(rotation_matrix, shear_matrix), tf.matmul(zoom_matrix, shift_matrix))


def transform_batch(images,
                    max_rot_deg,
                    max_shear_deg,
                    max_zoom_diff_pct,
                    max_shift_pct,
                    experimental_tpu_efficiency=True):
  """Transform a batch of square images with the same randomized affine
  transformation.
  """

  def clipped_random():
    rand = tf.random.normal([1], dtype=tf.float32)
    rand = tf.clip_by_value(rand, -2., 2.) / 2.
    return rand

  batch_size = images.shape[0]
  tf.debugging.assert_equal(
    images.shape[1],
    images.shape[2],
    "Images should be square")
  DIM = images.shape[1]
  channels = images.shape[3]
  XDIM = DIM % 2

  rot = max_rot_deg * clipped_random()
  shr = max_shear_deg * clipped_random() 
  h_zoom = 1.0 + clipped_random()*max_zoom_diff_pct
  w_zoom = 1.0 + clipped_random()*max_zoom_diff_pct
  h_shift = clipped_random()*(DIM*max_shift_pct)
  w_shift = clipped_random()*(DIM*max_shift_pct)

  # GET TRANSFORMATION MATRIX
  m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

  # LIST DESTINATION PIXEL INDICES
  x = tf.repeat(tf.range(DIM//2,-DIM//2,-1), DIM) # 10000,
  y = tf.tile(tf.range(-DIM//2,DIM//2),[DIM])
  z = tf.ones([DIM*DIM],tf.int32)
  idx = tf.stack( [x,y,z] ) # [3, 10000]

  # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
  idx2 = tf.matmul(m,tf.cast(idx,tf.float32))
  idx2 = tf.cast(idx2,tf.int32)
  idx2 = tf.clip_by_value(idx2,-DIM//2+XDIM+1,DIM//2)

  # FIND ORIGIN PIXEL VALUES           
  idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
  idx3 = tf.transpose(idx3)
  batched_idx3 = tf.tile(idx3[tf.newaxis], [batch_size, 1, 1])

  if experimental_tpu_efficiency:
    # This reduces excessive padding in the original tf.gather_nd op
    idx4 = idx3[:, 0] * DIM + idx3[:, 1]
    images = tf.reshape(images, [batch_size, DIM * DIM, channels])
    d = tf.gather(images, idx4, axis=1)
    return tf.reshape(d, [batch_size,DIM,DIM,channels])
  else:
    d = tf.gather_nd(images, batched_idx3, batch_dims=1)
    return tf.reshape(d,[batch_size,DIM,DIM,channels])


class DifferentiableAugment:
  """Collection of image augmentation functions implmented in pure TensorFlow,
  so that they are fully differentiable (gradients are not lost when applied.)
  
  Each augmentation function takes a tensor of images and a difficulty value
  from 0 to 15. For curriculum learning, the difficulty can be increased slowly
  over time.
  """


  @staticmethod
  def static(imgs, DIFFICULTY):
    """Gaussian noise, or "static"
    """
    STATIC_STDDEVS = [
      0.00,
      0.03,
      0.06,
      0.1,
      0.13,
      0.16,
      0.2,
      0.23,
      0.26,
      0.3,
      0.33,
      0.36,
      0.4,
      0.43,
      0.46,
      0.50,
    ]
    img_shape = imgs[0].shape
    batch_size = imgs.shape[0]
    stddev = tf.gather(STATIC_STDDEVS, DIFFICULTY)
    stddev = tf.random.uniform([], 0, stddev)
    noise = tf.random.normal((batch_size, *img_shape), mean=0, stddev=stddev)
    imgs = imgs + noise
    return imgs


  @staticmethod
  def blur(imgs, DIFFICULTY):
    """Apply blur via a Gaussian convolutional kernel
    """
    STDDEVS = [
      0.01,
      0.3,
      0.6,
      0.8,
      1.0,
      1.3,
      1.6,
      1.8,
      2.0,
      2.3,
      2.6,
      2.8,
      3.0,
      3.3,
      3.6,
      3.8
    ]
    img_shape = imgs[0].shape
    c = img_shape[2]
    stddev = tf.gather(STDDEVS, DIFFICULTY)
    stddev = tf.random.uniform([], 0, stddev)
    gauss_kernel = gaussian_k(7, 7, 3, 3, stddev)

    # expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    
    # convolve
    out_channels = []
    for c_ in range(c):
      in_channel = tf.expand_dims(imgs[..., c_], -1)
      out_channel = tf.nn.conv2d(in_channel, gauss_kernel, strides=1, padding="SAME")
      out_channel = out_channel[..., 0]
      out_channels.append(out_channel)
    imgs = tf.stack(out_channels, axis=-1)
    return imgs


  @staticmethod
  def random_scale(imgs, DIFFICULTY):
    """Randomly scales all of the values in each channel
    """
    MULTIPLY_SCALES = [
      [1, 1],
      [0.9, 1.1],
      [0.85, 1.15],
      [0.8, 1.2],
      [0.75, 1.25],
      [0.7, 1.3],
      [0.65, 1.325],
      [0.6, 1.35],
      [0.55, 1.375],
      [0.50, 1.4],
      [0.48, 1.42],
      [0.46, 1.44],
      [0.44, 1.46],
      [0.42, 1.48],
      [0.4, 1.5],
      [0.35, 1.6],
    ]
    channels = imgs.shape[-1]
    scales = tf.gather(MULTIPLY_SCALES, DIFFICULTY)
    scales = tf.random.uniform([channels], minval=scales[0], maxval=scales[1])
    imgs = imgs * scales
    return imgs


  @staticmethod
  def transform(imgs, DIFFICULTY):
    DEGREES = [
      0.,
      2.,
      4.,
      6.,
      8.,
      10.,
      12.,
      14.,
      16.,
      18.,
      20.,
      22.,
      24.,
      26.,
      28.,
      30.,
    ]
    RESIZE_SCALES = [
      0,
      0.1,
      0.15,
      0.2,
      0.25,
      0.3,
      0.325,
      0.35,
      0.375,
      0.4,
      0.42,
      0.44,
      0.46,
      0.48,
      0.5,
      0.5
    ]
    SHIFT_PERCENTS = [
      0.00,
      0.025,
      0.05,
      0.07,
      0.09,
      0.1,
      0.11,
      0.12,
      0.13,
      0.14,
      0.15,
      0.16,
      0.17,
      0.18,
      0.19,
      0.20
    ]
    max_rot_deg = tf.gather(DEGREES, DIFFICULTY)
    max_shear_deg = tf.gather(DEGREES, DIFFICULTY) / 2.
    max_zoom_diff_pct = tf.gather(RESIZE_SCALES, DIFFICULTY)
    max_shift_pct = tf.gather(SHIFT_PERCENTS, DIFFICULTY)
    return transform_batch(
      imgs,
      max_rot_deg,
      max_shear_deg,
      max_zoom_diff_pct,
      max_shift_pct)


def get_noisy_channel():
  """Return a function that adds noise to a batch of images, conditioned on a
  difficulty value.
  """
  def no_aug(images, DIFFICULTY):
    return images

  func_names=[
    'static',
    'blur',
    'random_scale',
    'transform'
  ]
  func_names = [n for n in func_names if n is not None]


  @tf.function
  def noise_pipeline(images, funcs, DIFFICULTY):
    """Apply a series of functions to images, in order
    """
    if DIFFICULTY == 0:
      return images
    else:
      for func in funcs:
        images = func(images, DIFFICULTY)
      return images
  funcs = []
  for func_name in func_names:
    assert func_name in dir(DifferentiableAugment), f"Function '{func_name}' doesn't exist"
    funcs.append(getattr(DifferentiableAugment, func_name))
  return lambda images, DIFFICULTY: noise_pipeline(images, funcs, DIFFICULTY)
