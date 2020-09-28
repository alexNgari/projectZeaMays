"""
"""

#%%
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import tensorboard
from tensorboard.plugins.hparams import api as hp
from src.preprocessing.image_gen import MultiTaskImageGen2, BalanceImageGenerator
from src.applications.resnet_custom import make_model

#%%
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)

#%%
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATADIR = 'gs://alexfyp_data'
LOGDIR = 'gs://alexfyp_logs'
BATCH_SIZE = 128
CLASS_LABELS = ['faw', 'zinc_def', 'nlb', 'healthy']

EPOCHS=100
num_nlb = 14570
STEPS_PER_EPOCH = np.ceil(3*0.8*0.8*num_nlb/BATCH_SIZE)

#%%
feature_description = {
    'rows': tf.io.FixedLenFeature([1], tf.int64),
    'cols': tf.io.FixedLenFeature([1], tf.int64),
    'channels': tf.io.FixedLenFeature([1], tf.int64),
    'image': tf.io.FixedLenFeature([1], tf.string),
    'labels': tf.io.VarLenFeature(tf.float32)
}

#%%
train_files = [os.path.join(DATADIR, f'sharded/train/train{index}.tfrecord') for index in range(16)]
test_files = [os.path.join(DATADIR, f'sharded/test/test{index}.tfrecord') for index in range(8)]
val_files = [os.path.join(DATADIR, f'sharded/val/val{index}.tfrecord') for index in range(8)]

#%%
train = MultiTaskImageGen2(train_files, feature_description)
val = MultiTaskImageGen2(val_files, feature_description)
test = MultiTaskImageGen2(test_files, feature_description)

#%%
train_ds = train.get_all().shuffle(2048).prefetch(AUTOTUNE)
val_ds = val.get_all().shuffle(1024).prefetch(AUTOTUNE)
test_ds = test.get_all().shuffle(1024)#.batch(BATCH_SIZE, drop_remainder=True)

#%%
# def train_generator():
#   for X, (y1, y2, y3) in train_ds:
#     yield X.numpy(), (y1.numpy(), y2.numpy(), y3.numpy())

# def val_generator():
#   for X, (y1, y2, y3) in val_ds:
#     yield X.numpy(), (y1.numpy(), y2.numpy(), y3.numpy())

# def test_generator():
#   for X, (y1, y2, y3) in test_ds:
#     yield X.numpy(), [y1.numpy(), y2.numpy(), y3.numpy()]

#%%
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
# train_ds = train_ds.with_options(options)
# val_ds = val_ds.with_options(options)
# test_ds = test_ds.with_options(options)

#%%
# with strategy.scope():
#   initializer = tf.keras.initializers.he_normal()
#   loss = tf.keras.losses.BinaryCrossentropy()
#   optimizer = tf.keras.optimizers.Adam()
#   METRICS = [tf.keras.metrics.BinaryAccuracy(name='acc'),
#               tf.keras.metrics.Precision(name='psn'),
#               tf.keras.metrics.Recall(name='rcl'),
#               tf.keras.metrics.AUC(name='AUC')]
#   model = make_model((256,256,3), METRICS, optimizer, loss, weights_initializer=initializer)

# model.summary

#%%
# model.fit(train_generator(),
#           epochs=EPOCHS,
#           steps_per_epoch=STEPS_PER_EPOCH,
#           validation_data=val_generator())

train_X = np.empty((train.get_num_images(), 256, 256, 3), dtype=np.float32)
train_y = np.empty((train.get_num_images(), 3), dtype=np.float32)
test_X = np.empty((test.get_num_images(), 256, 256, 3), dtype=np.float32)
test_y = np.empty((test.get_num_images(), 3), dtype=np.float32)
val_X = np.empty((val.get_num_images(), 256, 256, 3), dtype=np.float32)
val_y = np.empty((val.get_num_images(), 3), dtype=np.float32)

index = 0

for X, (y1, y2, y3) in train_ds:
  train_X[index] = X.numpy()
  train_y[index] = (y1.numpy(), y2.numpy(), y3.numpy())
  index += 1

index = 0
for X, (y1, y2, y3) in test_ds:
  test_X[index] = X.numpy()
  test_y[index] = (y1.numpy(), y2.numpy(), y3.numpy())
  index += 1

index = 0
for X, (y1, y2, y3) in val_ds:
  val_X[index] = X.numpy()
  val_y[index] = (y1.numpy(), y2.numpy(), y3.numpy())
  index += 1