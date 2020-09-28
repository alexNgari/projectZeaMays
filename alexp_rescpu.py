"""
Resnet
"""

#%% #Imports
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import tensorboard
from tensorboard.plugins.hparams import api as hp
from src.preprocessing.image_gen import MultiTaskImageGen2, BalanceImageGenerator
from src.applications.resnet_custom import make_model

#%% #Check for TPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

#%%
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clean')
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/alexp_resCPU/{TIMESTAMP}')
BATCH_SIZE = 16
CLASS_LABELS = ['FAW', 'zinc_def', 'healthy']

#%%
feature_description = {
    'rows': tf.io.FixedLenFeature([1], tf.int64),
    'cols': tf.io.FixedLenFeature([1], tf.int64),
    'channels': tf.io.FixedLenFeature([1], tf.int64),
    'image': tf.io.FixedLenFeature([1], tf.string),
    'labels': tf.io.VarLenFeature(tf.float32)
}

#%% #Dataset initialisation
ds_faw = MultiTaskImageGen2(os.path.join(DATADIR, 'final/faw.tfrecord'), feature_description)
test_faw, val_faw = ds_faw.split_dataset()
ds_healthy = MultiTaskImageGen2(os.path.join(DATADIR, 'final/healthy.tfrecord'), feature_description)
test_healthy, val_healthy = ds_healthy.split_dataset()
ds_zinc = MultiTaskImageGen2(os.path.join(DATADIR, 'final/zinc_def.tfrecord'), feature_description)
test_zinc, val_zinc = ds_zinc.split_dataset()

# ds_nlb = MultiTaskImageGen(os.path.join(DATADIR, 'final/NLB/nlb'), 256, CLASS_LABELS)
# ds_nlb_h = MultiTaskImageGen(os.path.join(DATADIR, 'final/NLB/nlb'), 256, CLASS_LABELS)

#%% #Test set
test = test_faw.concatenate(test_healthy)
test = test.concatenate(test_zinc).shuffle(1000)
test = test.batch(2*BATCH_SIZE)

#%% #Validation set
val = val_faw.concatenate(val_healthy)
val = val.concatenate(val_zinc)
val = val.batch(2*BATCH_SIZE)

#%%
num_healthy = len(os.listdir(os.path.join(DATADIR, 'final/healthy')))
STEPS_PER_EPOCH = np.ceil(3*0.8*0.8*num_healthy/BATCH_SIZE)
print(num_healthy, STEPS_PER_EPOCH)

#%%
balance_ds = BalanceImageGenerator(BATCH_SIZE, ds_faw(), ds_healthy(), ds_zinc())()

#%%
initializer = tf.keras.initializers.he_normal()
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
METRICS = [tf.keras.metrics.BinaryAccuracy(name='acc'),
            tf.keras.metrics.Precision(name='psn'),
            tf.keras.metrics.Recall(name='rcl'),
            tf.keras.metrics.AUC(name='AUC')]

#%%
# img_input = layers.Input(shape=(256,256,3))
# x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
# x = layers.Conv2D(64, (7, 7),
#                     strides=(2, 2),
#                     padding='valid',
#                     kernel_initializer='he_normal',
#                     name='conv1')(x)
# x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)
# x = layers.Activation('relu')(x)
# x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
# x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)

# faw = layers.Dense(1, activation='sigmoid', name='faw')(x)
# zinc = layers.Dense(1, activation='sigmoid', name='zinc')(x)
# # nlb = layers.Dense(1, activation='sigmoid', name='nlb')(x)

# model = tf.keras.Model(inputs=img_input, outputs=[faw, zinc])
# model.summary()

# #%%
# model.compile(optimizer=optimizer,
#             loss=loss,
#             metrics=METRICS)

# model = make_model((256,256,3), METRICS, optimizer, loss, initializer)
# model.summary()
# #%%
# model.fit(balance_ds,
#           epochs=5,
#           steps_per_epoch=STEPS_PER_EPOCH,
#           validation_data=val,
#           callbacks=[tf.keras.callbacks.TensorBoard(log_dir=LOGDIR, histogram_freq=1),
#                      tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# #%%
# # model.save("gs://eeefyp/models/resnet50", include_optimizer=False)

# #%% #Evaluate model
# model.evaluate(test)#, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)])

#%%
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/alexp_resCPU_extra/{TIMESTAMP}')

#%%
model = make_model((256,256,3), METRICS, optimizer, loss, initializer, True)
model.summary()
#%%
model.fit(balance_ds,
          epochs=5,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=val,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir=LOGDIR, histogram_freq=1),
                     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

#%%
# model.save("gs://eeefyp/models/resnet50", include_optimizer=False)

#%% #Evaluate model
model.evaluate(test)#, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)])

#%%