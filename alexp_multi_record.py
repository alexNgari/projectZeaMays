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
from src.preprocessing.image_gen import MultiTaskImageGen2, BalanceImageGenerator

#%%
tf.config.experimental.list_physical_devices('GPU')

#%%
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%%
# tf.config.optimizer.set_jit(True)
tf.config.optimizer.get_jit()

#%%
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clean')
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/custom_loop001')
BATCH_SIZE = 32
CLASS_LABELS = ['FAW', 'zinc_def', 'healthy']
EPOCHS = 10

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
img_ds = ds_faw.get_train_img()

ds_healthy = MultiTaskImageGen2(os.path.join(DATADIR, 'final/healthy.tfrecord'), feature_description)
test_healthy, val_healthy = ds_healthy.split_dataset()
img_ds = img_ds.concatenate(ds_healthy.get_train_img())

ds_zinc = MultiTaskImageGen2(os.path.join(DATADIR, 'final/zinc_def.tfrecord'), feature_description)
test_zinc, val_zinc = ds_zinc.split_dataset()
img_ds = img_ds.concatenate(ds_zinc.get_train_img())

# ds_nlb = MultiTaskImageGen(os.path.join(DATADIR, 'final/NLB/nlb'), 256, CLASS_LABELS)
# ds_nlb_h = MultiTaskImageGen(os.path.join(DATADIR, 'final/NLB/nlb'), 256, CLASS_LABELS)

#%% #Test set
test = test_faw.concatenate(test_healthy)
test = test.concatenate(test_zinc).shuffle(1000)
test = test.batch(BATCH_SIZE)

#%% #Validation set
val = val_faw.concatenate(val_healthy)
val = val.concatenate(val_zinc)
val = val.batch(BATCH_SIZE)

#%%
num_healthy = len(os.listdir(os.path.join(DATADIR, 'final/healthy')))
STEPS_PER_EPOCH = np.ceil(3*0.8*0.8*num_healthy/BATCH_SIZE)
print(num_healthy, STEPS_PER_EPOCH)

#%%
balance_ds = BalanceImageGenerator(BATCH_SIZE, ds_faw(), ds_healthy(), ds_zinc())()

#%%
mixed_precision_policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale='dynamic')
tf.keras.mixed_precision.experimental.set_policy(mixed_precision_policy)

#%%
inputs = Input(shape=(256, 256, 3))
norm = layers.experimental.preprocessing.Normalization()
norm.adapt(img_ds)
norm_layer = norm(inputs)
conv1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal')(norm_layer)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal')(pool1)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
conv3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal')(pool2)
flatten = layers.Flatten()(conv3)
dense1 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(flatten)
faw = layers.Dense(1, activation='sigmoid', name='faw')(dense1)
zinc = layers.Dense(1, activation='sigmoid', name='zinc')(dense1)
model = tf.keras.Model(inputs=inputs, outputs=[faw, zinc])

# model.summary()

# METRICS1 = [tf.keras.metrics.BinaryAccuracy(name='acc'),
#             tf.keras.metrics.Precision(name='psn'),
#             tf.keras.metrics.Recall(name='rcl'),
#             tf.keras.metrics.AUC(name='AUC')]

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

#%%
writer = tf.summary.create_file_writer(LOGDIR)

optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_acc')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_acc')

train_faw_loss = tf.keras.metrics.Mean(name='train_faw_loss')
val_faw_loss = tf.keras.metrics.Mean(name='val_faw_loss')
faw_metrics = [tf.keras.metrics.BinaryAccuracy(name='faw_acc'),
               tf.keras.metrics.Precision(name='faw_psn'),
               tf.keras.metrics.Recall(name='faw_rcl'),
               tf.keras.metrics.AUC(name='faw_AUC')]

train_zn_loss = tf.keras.metrics.Mean(name='train_zn_loss')
val_zn_loss = tf.keras.metrics.Mean(name='val_zn_loss')
zn_metrics = [tf.keras.metrics.BinaryAccuracy(name='zn_acc'),
              tf.keras.metrics.Precision(name='zn_psn'),
              tf.keras.metrics.Recall(name='zn_rcl'),
              tf.keras.metrics.AUC(name='zn_AUC')]

loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

def t_step():
    @tf.function
    def train_step(batch):
        images, labels = batch
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss_faw = compute_loss(labels[0], preds[0])
            loss_zn = compute_loss(labels[1], preds[1])
            loss_tot = tf.add(loss_faw, loss_zn)
        grads = tape.gradient(loss_tot, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss.update_state(loss_tot)
        train_faw_loss.update_state(loss_faw)
        train_zn_loss.update_state(loss_zn)
        train_accuracy.update_state(labels, preds)
        for metric in faw_metrics:
            metric.update_state(labels[0], preds[0])
        for metric in zn_metrics:
            metric.update_state(labels[1], preds[1])
        return loss_tot
    return train_step

def v_step():
    @tf.function
    def val_step(batch):
        images, labels = batch
        predictions = model(images, training=False)
        faw_loss = loss_object(labels[0], predictions[0])
        zn_loss = loss_object(labels[1], predictions[1])
        loss_tot = tf.add(faw_loss, zn_loss)
        val_loss.update_state(loss_tot)
        val_faw_loss.update_state(faw_loss)
        val_zn_loss.update_state(zn_loss)
        val_accuracy.update_state(labels, predictions)
        for metric in faw_metrics:
            metric.update_state(labels[0], predictions[0])
        for metric in zn_metrics:
            metric.update_state(labels[1], predictions[1])
        return loss_tot
    return val_step

#%%
with writer.as_default():
    for epoch in range(EPOCHS):
        for batch in balance_ds.take(STEPS_PER_EPOCH):
            train_step = t_step()
            train_step(batch)
        print(f'loss: {train_loss.result():.3f} Accuracy: {train_accuracy.result():.3f} FAW: {train_faw_loss.result():.3f} Zn: {train_zn_loss.result():.3f}')
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
        writer.flush()
        tf.summary.scalar('train_acc', train_accuracy.result(), step=epoch)
        writer.flush()
        tf.summary.scalar('train_faw_loss', train_faw_loss.result(), step=epoch)
        writer.flush()
        tf.summary.scalar('train_zinc_loss', train_zn_loss.result(), step=epoch)
        writer.flush()
        metrics = ['accuracy', 'precision', 'recall', 'AUC']
        for i, metric in enumerate(faw_metrics):
            tf.summary.scalar(f'faw{metrics[i]}', metric.result(), step=epoch)
            writer.flush()
        for i, metric in enumerate(zn_metrics):
            tf.summary.scalar(f'zn{metrics[i]}', metric.result(), step=epoch)
            writer.flush()


        for metric in faw_metrics:
            metric.reset_states()
        for metric in zn_metrics:
            metric.reset_states()

        for batch in val:
            val_step = v_step()
            val_step(batch)
        print(f'loss: {val_loss.result():.3f} Accuracy: {val_accuracy.result():.3f} FAW: {val_faw_loss.result():.3f} Zn: {val_zn_loss.result():.3f}')
        print(f'f_acc: {faw_metrics[0].result():.3f} f_psn: {faw_metrics[1].result():.3f} f_rcl: {faw_metrics[2].result():.3f} f_AUC: {faw_metrics[3].result():.3f}')
        print(f'z_acc: {zn_metrics[0].result():.3f} z_psn: {zn_metrics[1].result():.3f} z_rcl: {zn_metrics[2].result():.3f} z_AUC: {zn_metrics[3].result():.3f}')
        print('\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        tf.summary.scalar('val_loss', val_loss.result(), step=epoch)
        writer.flush()
        tf.summary.scalar('val_acc', val_accuracy.result(), step=epoch)
        writer.flush()
        tf.summary.scalar('val_faw_loss', val_faw_loss.result(), step=epoch)
        writer.flush()
        tf.summary.scalar('val_zinc_loss', val_zn_loss.result(), step=epoch)
        writer.flush()
        metrics = ['accuracy', 'precision', 'recall', 'AUC']
        for i, metric in enumerate(faw_metrics):
            tf.summary.scalar(f'faw{metrics[i]}', metric.result(), step=epoch)
            writer.flush()
        for i, metric in enumerate(zn_metrics):
            tf.summary.scalar(f'zn{metrics[i]}', metric.result(), step=epoch)
            writer.flush()

        train_loss.reset_states()
        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()
        train_faw_loss.reset_states()
        val_faw_loss.reset_states()
        train_zn_loss.reset_states()
        val_zn_loss.reset_states()
        for metric in faw_metrics:
            metric.reset_states()
        for metric in zn_metrics:
            metric.reset_states()
            

