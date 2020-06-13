import code
import secrets

import tensorflow as tf
import matplotlib.pyplot as plt

from cfg import get_config; CFG = get_config()
from data import get_dataset
from models import get_model, get_optim, get_loss_fn, get_metric_fn, \
  DifficultyManager

if __name__ == "__main__":
  ds, val_ds = get_dataset()
  model, loss_fn, metric_fn = \
    get_model(), get_loss_fn(), get_metric_fn()


  path_prefix = CFG['path_prefix']
  run_name = CFG['run_name'] + '_' + secrets.token_hex(2)
  tbc = tf.keras.callbacks.TensorBoard(log_dir=f"{path_prefix}logs/{run_name}")
  diffc = DifficultyManager()

  # build and compile model
  for val in ds.take(1): model(val)
  optim = get_optim(model)
  model.compile(optimizer=optim, loss_fn=loss_fn, metric_fn=metric_fn)
  model.run_eagerly = CFG['eager_mode']

  # train loop
  model.fit(x=ds,
            validation_data=val_ds,
            epochs=CFG['epochs'],
            callbacks=[tbc, diffc],
  )

  Z = model.symbol_embed(val)
  imgs = model.generator(Z)
  fig, axes = plt.subplots(2, 2)
  axes[0][0].imshow(imgs[0])
  axes[0][1].imshow(imgs[1])
  axes[1][0].imshow(imgs[2])
  axes[1][1].imshow(imgs[3])
  plt.show()