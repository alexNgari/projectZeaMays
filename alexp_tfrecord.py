"""
Convert dataset to tfrecord
"""
#%%
import os
import numpy as np
import tensorflow as tf
from src.preprocessing.image_gen import MultiTaskImageGen
from src.preprocessing.record_dataset import GenerateTFRecord
from src.preprocessing.dataset_stuff import decode_image

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
#%% #Define constants
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clean')
CLASS_LABELS = ['faw', 'zinc_def', 'nlb', 'healthy']
CLASS_LABELS

#%% #Write datasets
t = GenerateTFRecord(CLASS_LABELS)
t.convert_image_folder(os.path.join(DATADIR, 'final/healthy'), os.path.join(DATADIR,'final/healthy.tfrecord'))
t.convert_image_folder(os.path.join(DATADIR, 'final/faw'), os.path.join(DATADIR,'final/faw.tfrecord'))
t.convert_image_folder(os.path.join(DATADIR, 'final/zinc_def'), os.path.join(DATADIR,'final/zinc_def.tfrecord'))

#%% #Read dataset
dataset = tf.data.TFRecordDataset(filenames=os.path.join(DATADIR, 'final/zinc_def.tfrecord'))

# %%
feature_description = {
    'rows': tf.io.FixedLenFeature([1], tf.int64),
    'cols': tf.io.FixedLenFeature([1], tf.int64),
    'channels': tf.io.FixedLenFeature([1], tf.int64),
    'image': tf.io.FixedLenFeature([1], tf.string),
    'labels': tf.io.VarLenFeature(tf.float32)
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

# %%
def _extract_fn(sample):
    image = decode_image(sample['image'][0])
    labels = tf.sparse.to_dense(sample['labels'])
    labels = labels[0], labels[1]
    return image, labels

#%%
parsed = dataset.map(_parse_function).map(_extract_fn)


# %%
