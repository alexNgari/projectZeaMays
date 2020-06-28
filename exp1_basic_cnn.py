"""
Stuff
"""
# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from src.preprocessing.image_gen import MultiTaskImageGen, BalanceImageGenerator

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data/clean/final')
TESTDIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data/clean/final_test')
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
BATCH_SIZE = 16
CLASS_LABELS = ['faw', 'zinc_def', 'healthy']
CLASS_LABELS


# %% #Dataset initialisation
ds_faw = MultiTaskImageGen(os.path.join(DATADIR, 'faw'), 256, CLASS_LABELS)
val_faw = ds_faw.split_dataset()
ds_healthy = MultiTaskImageGen(os.path.join(
    DATADIR, 'healthy'), 256, CLASS_LABELS)
val_healthy = ds_healthy.split_dataset()
ds_zinc = MultiTaskImageGen(os.path.join(
    DATADIR, 'zinc_def'), 256, CLASS_LABELS)
val_zinc = ds_zinc.split_dataset()

# %%
test_faw = MultiTaskImageGen(os.path.join(
    TESTDIR, 'faw'), 256, CLASS_LABELS).get_all()
test_healthy = MultiTaskImageGen(os.path.join(
    TESTDIR, 'healthy'), 256, CLASS_LABELS).get_all()
test_zinc = MultiTaskImageGen(os.path.join(
    TESTDIR, 'zinc_def'), 256, CLASS_LABELS).get_all()

# %%
for item, label in test_faw.take(1):
    # plt.imshow(item)
    print(label)

# %%
for item, label in val_faw.take(1):
    # plt.imshow(item)
    print(label)

# %%
for item, label in ds_faw().take(1):
    # plt.imshow(item)
    print(label)
print(ds_faw())

# %% #Test set
test = test_faw.concatenate(test_healthy)
test = test.concatenate(test_zinc).shuffle(1000)
test = test.batch(2*BATCH_SIZE)

# %%
for item in test.take(1):
    print(item)

# %% #Validation set
val = val_faw.concatenate(val_healthy)
val = val.concatenate(val_zinc)
val = val.batch(2*BATCH_SIZE)

# %%
num_healthy = len(os.listdir(os.path.join(DATADIR, 'healthy')))
STEPS_PER_EPOCH = np.ceil(3*0.8*num_healthy/BATCH_SIZE)
print(num_healthy, STEPS_PER_EPOCH)
EPOCHS = 200

# %%
balance_ds = BalanceImageGenerator(
    BATCH_SIZE, ds_faw(), ds_healthy(), ds_zinc())()

# %%
for item in balance_ds.take(1):
    print(item)

# %%
print(balance_ds)
print(val)

# %%
initializer = tf.keras.initializers.he_normal()
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

METRICS = [tf.keras.metrics.BinaryAccuracy(name='acc'),
           tf.keras.metrics.Precision(name='psn'),
           tf.keras.metrics.Recall(name='rcl'),
           tf.keras.metrics.AUC(name='AUC')]

inputs = Input(shape=(256, 256, 3))
conv1 = layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_initializer='he_normal')(inputs)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_initializer='he_normal')(pool1)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
conv3 = layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_initializer='he_normal')(pool2)
flatten = layers.Flatten()(conv3)
dense1 = layers.Dense(64, activation='relu',
                      kernel_initializer='he_normal')(flatten)
faw = layers.Dense(1, activation='sigmoid', name='faw')(dense1)
zinc = layers.Dense(1, activation='sigmoid', name='zinc')(dense1)
model = tf.keras.Model(inputs=inputs, outputs=[faw, zinc])

model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

model.summary()

# %%
logdir = os.path.join(LOGDIR, 'a-corrections_CNN_trial')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/ngari/Dev/projectzeamays/models/a-corrections_CNN',
                                                monitor='val_loss', verbose=0, save_best_only=True,
                                                save_weights_only=False, mode='auto', save_freq='epoch')
log_tens = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

model.fit(balance_ds,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=val,
          callbacks=[reduce_lr, checkpoint, log_tens])
# %%
history = model.evaluate(test)
history

# %%
new_model = tf.keras.models.load_model('/home/ngari/Dev/projectzeamays/models/a-corrections_CNN')
new_hist = new_model.evaluate(test)
new_hist

# %%
