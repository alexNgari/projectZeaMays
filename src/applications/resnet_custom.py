"""
Resnet architecture with 3 output nodes for transfer learning
"""

#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

#%%
class ResNetBlock(tf.keras.Model):
    """A ResNet identity block that can be appended to any architecture."""
    def __init__(self, kernel_size, filters, inilializer='he_normal'):
        super(ResNetBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters
        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), activation='relu',\
                                             kernel_initializer=inilializer)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same',\
                                             activation='relu', kernel_initializer=inilializer)
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1), activation='relu',\
                                             kernel_initializer=inilializer)
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
def make_model(input_shape, metrics, optimizer, loss, dropout, input_normalizer=None,\
                weights_initializer='he_normal', extra_layers=False):
    """
    Build and compiled model with resnet50 as the base model.

    # Args
        input_shape (tuple)
        metrics (list)
        optimizer (tf.keras.optimizers.Optimizer)
        loss (tf.keras.losses.Loss)
        input_normalizer (tf.keras.layers.Layer, optional): BatchNorm or prep.Norm
        weights_initializer (tf.keras.initializers.Initializer, optional)
        extra_layers (bool, optional): whether to include res block before O/P nodes

    # Returns
        tf.keras.Model: a compiled keras model
    """
    
    res_block = ResNetBlock((3,3), [512,512,2048], weights_initializer)
    resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape,\
                                            weights=None)
    for layer in resnet.layers:
        if hasattr(layer, 'kernel_initializer'):
            setattr(layer, 'kernel_initializer', weights_initializer)
    
    inputs = Input(shape=input_shape)
    if input_normalizer:
        x = input_normalizer(inputs)
        x = resnet(x)
    else:
        x = resnet(inputs)    
    
    if extra_layers:
        faw = layers.Conv2D(512, (1,1), activation='relu', kernel_initializer=weights_initializer)(x)
        faw = layers.BatchNormalization()(faw)
        faw = layers.Activation('relu')(faw)
        faw = layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer=weights_initializer)(faw)
        faw = layers.BatchNormalization()(faw)
        faw = layers.Activation('relu')(faw)
        faw = layers.Conv2D(2048, (1,1), activation='relu', kernel_initializer=weights_initializer)(faw)
        faw = layers.BatchNormalization()(faw)
        faw = layers.Add()([faw, x])
        faw = layers.Activation('relu')(faw)
        faw = layers.GlobalAveragePooling2D()(faw)
        faw = layers.Dropout(dropout)(faw)
        faw = layers.Dense(1, activation='sigmoid', name='faw', dtype='float32')(faw)

        zinc = layers.Conv2D(512, (1,1), activation='relu', kernel_initializer=weights_initializer)(x)
        zinc = layers.BatchNormalization()(zinc)
        zinc = layers.Activation('relu')(zinc)
        zinc = layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer=weights_initializer)(zinc)
        zinc = layers.BatchNormalization()(zinc)
        zinc = layers.Activation('relu')(zinc)
        zinc = layers.Conv2D(2048, (1,1), activation='relu', kernel_initializer=weights_initializer)(zinc)
        zinc = layers.BatchNormalization()(zinc)
        zinc = layers.Add()([zinc, x])
        zinc = layers.Activation('relu')(zinc)
        zinc = layers.GlobalAveragePooling2D()(zinc)
        zinc = layers.Dropout(dropout)(zinc)
        zinc = layers.Dense(1, activation='sigmoid', name='zinc', dtype='float32')(zinc)

        nlb = layers.Conv2D(512, (1,1), activation='relu', kernel_initializer=weights_initializer)(x)
        nlb = layers.BatchNormalization()(nlb)
        nlb = layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer=weights_initializer)(nlb)
        nlb = layers.BatchNormalization()(nlb)
        nlb = layers.Conv2D(2048, (1,1), activation='relu', kernel_initializer=weights_initializer)(nlb)
        nlb = layers.BatchNormalization()(nlb)
        nlb = layers.Add()([nlb, x])
        nlb = layers.Activation('relu')(nlb)
        nlb = layers.GlobalAveragePooling2D()(nlb)
        nlb = layers.Dropout(dropout)(nlb)
        nlb = layers.Dense(1, activation='sigmoid', name='nlb', dtype='float32')(nlb)
    
    else:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
        faw = layers.Dense(1, activation='sigmoid', name='faw', dtype='float32')(x)
        zinc = layers.Dense(1, activation='sigmoid', name='zinc', dtype='float32')(x)
        nlb = layers.Dense(1, activation='sigmoid', name='nlb', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[faw, zinc, nlb])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

