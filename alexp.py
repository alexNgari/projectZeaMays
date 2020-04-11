"""
Stuff
"""
#%%
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorboard
from tensorboard.plugins.hparams import api as hp
from src.preprocessing.image_gen import ImageGenerator, BalanceImageGenerator
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
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clean/final')
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/alexp/{TIMESTAMP}')
BATCH_SIZE = 16
CLASS_LABELS = os.listdir(DATADIR)
CLASS_LABELS

#%% #Dataset initialisation
ds_faw = ImageGenerator(os.path.join(DATADIR, 'FAW'), 256, CLASS_LABELS)
test_faw, val_faw = ds_faw.split_dataset()
ds_healthy = ImageGenerator(os.path.join(DATADIR, 'healthy'), 256, CLASS_LABELS)
test_healthy, val_healthy = ds_healthy.split_dataset()
ds_zinc = ImageGenerator(os.path.join(DATADIR, 'zinc_def'), 256, CLASS_LABELS)
test_zinc, val_zinc = ds_zinc.split_dataset()

#%% #Test set
test = test_faw.concatenate(test_healthy)
test = test.concatenate(test_zinc).shuffle(1000)
test = test.batch(2*BATCH_SIZE)

#%% #Validation set
val = val_faw.concatenate(val_healthy)
val = val.concatenate(val_zinc)
val = val.batch(2*BATCH_SIZE)

#%%
num_healthy = len(os.listdir(os.path.join(DATADIR, 'healthy')))
STEPS_PER_EPOCH = np.ceil(3*0.8*0.8*num_healthy/BATCH_SIZE)
print(num_healthy, STEPS_PER_EPOCH)

#%%
balance_ds = BalanceImageGenerator(BATCH_SIZE, ds_faw(), ds_healthy(), ds_zinc())()

#%%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))

model.summary()
#%%
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dense(3, activation='softmax', name='local'))

model.summary()

#%%
METRICS = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Recall(class_id=2, name='zn_recall'),
            tf.keras.metrics.AUC(name='AUC'),
            f1_metrics.F1Score(3, name='f1_score')]
# %%
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=METRICS)

#%%
log_dir = os.path.join(LOGDIR, 'train')
history = model.fit(balance_ds, epochs=5,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=val)
                    # callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])



# %%
results = model.evaluate(test)