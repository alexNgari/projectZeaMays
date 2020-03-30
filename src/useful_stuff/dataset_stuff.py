"""
Helper functions for the image generator classes.
"""
import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_all_files(data_directory):
    """
    param:  path to dataset
    return: Dataset object with all images in path
    """
    return tf.data.Dataset.list_files(f'{data_directory}/*')

# def get_label(file_path):
#     """
#     param:  path to an image
#     returns: the label of the image
#     """
#     path_sections = tf.strings.split(file_path, os.path.sep)
#     return path_sections[-2] # the label is the directory of the image

def decode_image(img):
    """
    param:  byte image
    return: image normalised in range [0,1]
    """
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def process_path(file_path):
    """
    param:  path to image
    returns: decoded image, label
    """
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    # label = get_label(file_path)
    return img#, label

def convert_dataset(paths_dataset):
    """
    param:  Dataset object with file paths to all images
    return: Fully processed dataset with image, label pairs
    """
    return paths_dataset.map(process_path, num_parallel_calls=AUTOTUNE)

def augment(img, label):
    """
    Add random perturbations to the images to reduce overfitting.
    Also SMOTE
    """
    # img = tf.image.random_brightness(img, 0.05)
    # img = tf.image.random_contrast(img, lower=0.0, upper=0.1)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # img = tf.image.random_jpeg_quality(img, min_jpeg_quality=70, max_jpeg_quality=100)
    # img = tf.image.random_hue(img, 0.3)
    # img = tf.image.random_saturation(img, lower=0.0, upper=0.3)
    return img, label
