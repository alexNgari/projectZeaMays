"""
Resnet
"""

#%% #Imports
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import tensorboard
from tensorboard.plugins.hparams import api as hp
from src.preprocessing.image_gen import MultiTaskImageGen, BalanceImageGenerator
from src.applications.resnet_custom import make_model

#%% #Check for TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='fyp',
                                                          zone='europe-west4-a',
                                                          project='eeefyp')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy=tf.distribute.experimental.TPUStrategy(resolver)                                                          
# try:
  # resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='fyp',
  #                                                         zone='europe-west4-a',
  #                                                         project='eeefyp') # TPU detection
# except ValueError: # If TPU not found
#   tpu = None 

# if tpu:
#   tf.config.experimental_connect_to_cluster(resolver)
#   tf.tpu.experimental.initialize_tpu_system(tpu)
#   strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128)
#   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
# else:
#   strategy = tf.distribute.get_strategy() # Default strategy that works on CPU and single GPU
#   print('Running default instead')
# print("Number of accelerators: ", strategy.num_replicas_in_sync)

#%% #Global vsrs
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATADIR = 'gs://eeefyp_data/clean'
LOGDIR = os.path.join('gs://eeefyp_logs', f'logs/resnet_trials/{TIMESTAMP}')
BATCH_SIZE = 32
CLASS_LABELS = ['faw', 'zinc_def', 'nlb', 'healthy']
CLASS_LABELS

#%% #Dataset initialisation
ds_faw = MultiTaskImageGen(os.path.join(DATADIR, 'final/faw'), 256, CLASS_LABELS)
test_faw, val_faw = ds_faw.split_dataset()

ds_zinc = MultiTaskImageGen(os.path.join(DATADIR, 'final/zinc_def'), 256, CLASS_LABELS)
test_zinc, val_zinc = ds_zinc.split_dataset()

ds_nlb = MultiTaskImageGen(os.path.join(DATADIR, 'NLB/nlb'), 256, CLASS_LABELS)
test_nlb, val_nlb = ds_nlb.split_dataset()

ds_healthy = MultiTaskImageGen(os.path.join(DATADIR, 'NLB/healthy'), 256, CLASS_LABELS)
test_healthy, val_healthy = ds_healthy.split_dataset()

#%% #Test set
test = test_faw.concatenate(test_healthy)
test = test.concatenate(test_zinc)
test = test.concatenate(test_nlb).shuffle(1000)
test = test.batch(2*BATCH_SIZE)

#%% #Validation set
val = val_faw.concatenate(val_healthy)
val = val.concatenate(val_zinc)
val = val.concatenate(val_healthy).shuffle(1000).cache()
val = val.batch(2*BATCH_SIZE)

#%%
num_nlb = 14570
STEPS_PER_EPOCH = np.ceil(3*0.8*0.8*num_nlb/BATCH_SIZE)
print(num_nlb, STEPS_PER_EPOCH)

#%%
balance_ds = BalanceImageGenerator(BATCH_SIZE, ds_faw(), ds_healthy(), ds_zinc(), ds_nlb())()

#%%
initializer = tf.keras.initializers.he_normal()
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
METRICS = [tf.keras.metrics.BinaryAccuracy(name='acc'),
            tf.keras.metrics.Precision(name='psn'),
            tf.keras.metrics.Recall(name='rcl'),
            tf.keras.metrics.AUC(name='AUC')]

#%%
with strategy.scope():
    model = make_model((256,256,3), METRICS, optimizer, loss, initializer)

#%%
model.fit(balance_ds,
          epochs=100,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=val,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir=LOGDIR, histogram_freq=1),
                     tf.keras.callbacks.EarlyStopping(monitor='val_nlb_loss', patience=5, restore_best_weights=True)])

#%%
model.save("gs://eeefyp/models/resnet50", include_optimizer=False)

#%% #Evaluate model
model.evaluate(test, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)])