"""
Helper functions for the image generator classes.
"""
import numpy as np
import tensorflow as tf
# import cv2
# from skimage import io, feature, color, img_as_ubyte

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

def tensor_to_feature(tensor: tf.Tensor):
    """
    Convert non-scalar tensor to a tf.train.Feature appendable in a tf.train.Example
    """
    serialized_tensor = tf.io.serialize_tensor(tensor)
    bytes_string = serialized_tensor.numpy()
    bytes_list = tf.train.BytesList(value=[bytes_string])
    return tf.train.Feature(bytes_list=bytes_list)


def create_example(img: tf.Tensor, lbl: tf.Tensor):
    """
    Converts an image, label pair to a tf.train.Example object
    """    
    feature = {
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[0]])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[1]])),
        'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[2]])),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=tf.unstack(lbl))),
        'image_raw': tensor_to_feature(img)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def separate_images(image, label):
    return image

# def get_colour_features(image):
#     """
#     param:  Abs path to image to extract features from.
#     return: Array of means and standard deviations of BGR.
#     """
#     image = cv2.imread(image)
#     (means, std_devs) = cv2.meanStdDev(image)
#     return np.concatenate([means, std_devs]).flatten()

# def get_texture_features(image):
#     """
#     param:  path to image
#     return: texture features: []
#     """
#     image = io.imread(image)
#     image = color.rgb2gray(image)
#     image = img_as_ubyte(image)
#     glcm = feature.greycomatrix(image, [1], [0])
#     features = []
#     features.append(feature.greycoprops(glcm, 'contrast')[0, 0])
#     features.append(feature.greycoprops(glcm, 'dissimilarity')[0, 0])
#     features.append(feature.greycoprops(glcm, 'homogeneity')[0, 0])
#     features.append(feature.greycoprops(glcm, 'energy')[0, 0])
#     features.append(feature.greycoprops(glcm, 'correlation')[0, 0])
#     features.append(feature.greycoprops(glcm, 'ASM')[0, 0])
#     return np.array(features)
