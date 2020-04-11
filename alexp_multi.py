"""
Stuff
"""
#%%
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import tensorboard
from tensorboard.plugins.hparams import api as hp
from src.preprocessing.image_gen import MultiTaskImageGen, BalanceImageGenerator
from src.metrics import f1_metrics

#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
#%%
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clean')
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/alexp_multi/{TIMESTAMP}')
BATCH_SIZE = 16
CLASS_LABELS = ['FAW', 'zinc_def', 'healthy']
CLASS_LABELS

#%% #Dataset initialisation
ds_faw = MultiTaskImageGen(os.path.join(DATADIR, 'final/FAW'), 256, CLASS_LABELS)
test_faw, val_faw = ds_faw.split_dataset()
ds_healthy = MultiTaskImageGen(os.path.join(DATADIR, 'final/healthy'), 256, CLASS_LABELS)
test_healthy, val_healthy = ds_healthy.split_dataset()
ds_zinc = MultiTaskImageGen(os.path.join(DATADIR, 'final/zinc_def'), 256, CLASS_LABELS)
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
inputs = Input(shape=(256, 256, 3))
conv1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal')(inputs)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal')(pool1)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
conv3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal')(pool2)
flatten = layers.Flatten()(conv3)
dense1 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(flatten)
faw = layers.Dense(1, activation='sigmoid', name='faw')(dense1)
zinc = layers.Dense(1, activation='sigmoid', name='zinc')(dense1)
model = tf.keras.Model(inputs=inputs, outputs=[faw, zinc])

model.summary()
#%%
METRICS1 = [tf.keras.metrics.BinaryAccuracy(name='acc'),
            tf.keras.metrics.Precision(name='psn'),
            tf.keras.metrics.Recall(name='rcl'),
            tf.keras.metrics.AUC(name='AUC')]
# METRICS2 = ['accuracy', tf.keras.metrics.AUC()]
# %%
# losses = {'local': tf.keras.losses.BinaryCrossentropy(),
#           'nlb': tf.keras.losses.BinaryCrossentropy()}
# metrics = {'local': METRICS1, 'nlb': METRICS2}
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS1)

#%%
log_dir = os.path.join(LOGDIR, 'train')
history = model.fit(balance_ds, epochs=5,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=val,
                    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])



# %%
results = model.evaluate(test)

# %%
