import code

import tensorflow as tf

from cfg import get_config; CFG = get_config()
from data import get_dataset
from models import get_model, get_optim, get_loss_fn, get_metric_fn

if __name__ == "__main__":
  ds, val_ds = get_dataset()
  model, optim, loss_fn, metric_fn = \
    get_model(), get_optim(), get_loss_fn(), get_metric_fn()

  path_prefix = CFG['path_prefix']
  tbc = tf.keras.callbacks.TensorBoard(log_dir=f"{path_prefix}logs")

  model.compile(optimizer=optim, loss_fn=loss_fn, metric_fn=metric_fn)
  model.run_eagerly = CFG['eager_mode']
  model.fit(x=ds,
            validation_data=val_ds,
            epochs=CFG['epochs'],
            callbacks=[tbc],
  )