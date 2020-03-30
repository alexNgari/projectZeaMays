"""
Stuff
"""
#%%
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from src.preprocessing.image_gen import ImageGenerator, BalanceImageGenerator

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
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clean/final')
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

#%%
for item, label in test_faw.take(1):
    # plt.imshow(item)
    print(label.numpy())

#%%
for item, label in val_faw.take(1):
    # plt.imshow(item)
    print(label)

#%%
for item, label in ds_faw().take(1):
    # plt.imshow(item)
    print(label)
print(ds_faw())

#%%
test = test_faw.concatenate(test_healthy)
test = test.concatenate(test_zinc).shuffle(1000)
test = test.batch(2*BATCH_SIZE)

#%%
for item in test.take(1):
    print(item)

#%%
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
for item in balance_ds.take(1):
    print(item)

#%%
print(balance_ds)
print(val)
# #%%
# def plot_examples(images, labels):
#     """
#     params: minibatches of images and their labels
#     output: a plot of 25 images with their labels as titles
#     """
#     plt.figure()
#     for i in range(9):
#         plot = plt.subplot(3, 3, i+1)
#         plot.imshow(images[i])
#         plot.axis('off')

#%%
# plot_examples(ds_faw().take(30))
#%%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()
#%%
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3))

model.summary()

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%%
history = model.fit(balance_ds, epochs=10,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=val)

#%%
# def make_model():
#     model = tf.keras.Sequential([
#         layers.Flatten(input_shape=(256, 256, 1)),
#         layers.Dense(4096, activation='relu'),
#         layers.Dense(4096, activation='relu'),
#         layers.Dense(3)
#         ])
#     model.compile(optimizer='adam',
#                   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#     return model

# #%%
# model_with_aug = make_model()
# model_with_aug.summary()

# #%%
# aug_history = model_with_aug.fit(balance_ds, epochs=50, validation_data=val)
# # %%
