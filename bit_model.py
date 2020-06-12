"""BiT Big Transfer
Using the ResNet-50x3 model from the paper.
It is pretrainedo n ImageNet-21k

wget https://storage.googleapis.com/bit_models/BiT-M-R50x3.h5
"""

import os

import tensorflow as tf

from bit.bit_resnet import ResnetV2

KNOWN_MODELS = {
  f'{bit}-R{l}x{w}': f'gs://bit_models/{bit}-R{l}x{w}.h5'
  for bit in ['BiT-S', 'BiT-M']
  for l, w in [(50, 1), (50, 3), (101, 1), (101, 3), (152, 4)]
}

NUM_UNITS = {
  k: (3, 4, 6, 3) if 'R50' in k else
  (3, 4, 23, 3) if 'R101' in k else
  (3, 8, 36, 3)
  for k in KNOWN_MODELS
}


def get_model(model_name='BiT-M-R50x1'):
  num_units = NUM_UNITS[model_name]
  filters_factor = int(model_name[-1])*4
  model = ResnetV2(
    num_units=num_units,
    num_outputs=21843,
    filters_factor=filters_factor,
    name="bit",
    trainable=True,
    dtype=tf.float32
  )
  model.build((None, None, None, 3))
  model_path = os.path.join('models', f"{model_name}.h5")
  model.load_weights(model_path)
  return model


if __name__ == "__main__":
  model = get_model()
