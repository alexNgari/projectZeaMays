"""
Resnet architecture with 3 output nodes for transfer learning
"""

#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

#%%
class ResNetBlock(tf.keras.Model):
    """
    A ResNet identity block that can be appended to any architecture.
    """
    def __init__(self, kernel_size, filters, inilializer='he_normal'):
        super(ResNetBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), activation='relu', kernel_initializer=inilializer)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', activation='relu', kernel_initializer=inilializer)
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1), activation='relu', kernel_initializer=inilializer)
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

#%%
def make_model(input_shape, metrics, optimizer, loss, weights_initializer='he_normal', extra_layers=False):
    res_block = ResNetBlock((3,3), [32,32,64], weights_initializer)
    resnet = tf.keras.applications.ResNet50(include_top=False, weights=None)
    for layer in resnet.layers:
        if hasattr(layer, 'kernel_initializer'):
            setattr(layer, 'kernel_initializer', weights_initializer)
    inputs = Input(shape=input_shape)
    x = resnet(inputs)
    if extra_layers:
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=weights_initializer)(x)
        x = res_block(x)
    x = layers.Flatten()(x)
    faw = layers.Dense(1, activation='sigmoid', name='faw')(x)
    zinc = layers.Dense(1, activation='sigmoid', name='zinc')(x)
    nlb = layers.Dense(1, activation='sigmoid', name='nlb')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[faw, zinc, nlb])

    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    return model

