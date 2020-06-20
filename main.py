import code
import secrets
import os

import tensorflow as tf
import matplotlib.pyplot as plt

from cfg import get_config; CFG = get_config()
from data import get_dataset
from models import get_model, get_optim, get_loss_fn, get_metric_fn, \
  DifficultyManager, ImageSnapshotManager, LearningRateManager, CheckpointSaver


def main():
  strategy = None
  if CFG['TPU']:
    TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    strategy = tf.distribute.get_strategy()

  num_replicas = strategy.num_replicas_in_sync

  with strategy.scope():
    ds, val_ds = get_dataset(num_replicas)

    model, (loss_fn, loss_object), metric_fn = \
      get_model(), get_loss_fn(), get_metric_fn()

    path_prefix = CFG['path_prefix']
    run_name = CFG['run_name']

    # callbacks
    logdir = f"{path_prefix}logs/{run_name}"
    tbc = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    snapc = ImageSnapshotManager(log_dir=logdir)
    diffc = DifficultyManager()
    lrmc = LearningRateManager()
    ckptc = CheckpointSaver()
    callbacks = [tbc, snapc, diffc, lrmc, ckptc]

    # build and compile model
    # (have to run inference first to populate variables)
    for val in ds.take(1):
      if CFG['full_model'] in ['vision']:
        val = val[:CFG['batch_size']]
      else:
        for key, value in val.items():
          val[key] = value[:CFG['batch_size']]
      model(val)
    print("FINISHED DEBUG INPUT")

    optim = get_optim(model)
    model.compile(optimizer=optim,
                  loss_fn=loss_fn,
                  loss_object=loss_object,
                  metric_fn=metric_fn,
                  num_replicas=num_replicas)
    model.run_eagerly = CFG['eager_mode']

    # train loop
    model.fit(x=ds,
              validation_data=val_ds,
              epochs=CFG['epochs'],
              callbacks=callbacks)


if __name__ == "__main__":
  main()
