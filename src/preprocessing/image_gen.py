"""
Methods and classes to generate batches of images to feed to models.
"""
#%% Imports
# import os
import tensorflow as tf
from src.preprocessing.dataset_stuff import get_all_files, process_path,\
                                            augment, decode_image, separate_images

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ImageGenerator():
    """
    Generator to produce batches of augmented data.
    """
    def __init__(self, data_dir, image_size, class_labels: list, *, for_cnn=True):
        """
        params: directory to images belonging to one class (str)
                integer describing number of pixels on one side of images.
        """
        self.for_cnn = for_cnn
        self.data_dir = data_dir
        self.label = data_dir.split('/')[-1]
        self.image_size = image_size
        self.class_labels: tf.Tensor = tf.convert_to_tensor(class_labels)
        self.indices = tf.range(self.class_labels.shape[0])
        self.full_set = None

    def encode_labels(self, img):
        """
        params: encoded image and string label
        return: image and one-hot-encoded label
        """
        encoded_label = tf.reshape(tf.where(self.class_labels==self.label), (1,))[0]
        encoded_label = tf.one_hot(self.indices, self.indices.shape[0])[encoded_label]
        # encoded_label = tf.convert_to_tensor(encoded_label, dtype=tf.int64)
        return img, encoded_label

    def get_num_images(self):
        """
        Get the number of images in the directory
        """
        return len(tf.io.gfile.listdir(self.data_dir))

    def split_dataset(self):
        """
        Get training and validation sets: splits 80-20 twice
        """
        self.full_set = get_all_files(self.data_dir)\
            .map(process_path, num_parallel_calls=AUTOTUNE)\
            .map(self.encode_labels, num_parallel_calls=AUTOTUNE)\
            .shuffle(self.get_num_images())
        test_set = self.full_set.take(int(0.2*self.get_num_images()))
        self.full_set = self.full_set.skip(int(0.2*self.get_num_images()))
        val_set = self.full_set.take(int(0.2*0.8*self.get_num_images()))
        self.full_set = self.full_set.skip(int(0.2*0.8*self.get_num_images()))
        return test_set, val_set

    def get_all(self):
        self.full_set = get_all_files(self.data_dir)\
            .map(process_path, num_parallel_calls=AUTOTUNE)\
            .map(self.encode_labels, num_parallel_calls=AUTOTUNE)
            # .shuffle(self.get_num_images())
        return self.full_set

    def get_train_img(self):
        return self.full_set.map(separate_images, num_parallel_calls=AUTOTUNE)

    def __call__(self):
        if self.for_cnn:
            return self.full_set\
                        .cache()\
                        .shuffle(int(0.8*0.8*self.get_num_images()))\
                        .map(augment, num_parallel_calls=AUTOTUNE)\
                        .repeat()
        # else:
        #     return self.full_set\
        #                 .shuffle(int(0.8*0.8*self.get_num_images()))\
        #                 .map(augment, num_parallel_calls=AUTOTUNE)\
        #                 .repeat()


class BalanceImageGenerator():
    """
    Generate balanced batches from balanced or imbalanced class dataset.
    Requires you initialise individual datasets for each of the classes,
    and pass them as arguments during initialisation.
    """
    def __init__(self, batch_size, *args):
        self.batch_size = batch_size
        self.datasets = [*args]

    def calculate_weights(self):
        """
        calculate equal weightings for all classes.
        """
        return [1/len(self.datasets) for i in self.datasets]

    def __call__(self):
        resampled_ds = tf.data.experimental.sample_from_datasets(self.datasets,\
            weights=self.calculate_weights())#\
            # .cache()\
            # .prefetch(AUTOTUNE)
            # .batch(self.batch_size)\
        return resampled_ds


class MultiTaskImageGen(ImageGenerator):
    """
    Generator to produce batches of augmented data, modified to work for multi-task learning
    """
    def __init__(self, data_dir, image_size, class_labels: list, *, for_cnn=True):
        """
        params: directory to images belonging to one class (str)
                integer describing number of pixels on one side of images.
        """
        self.for_cnn = for_cnn
        self.data_dir = data_dir
        self.label = data_dir.split('/')[-1]
        self.image_size = image_size
        self.class_labels: tf.Tensor = tf.convert_to_tensor(class_labels)
        self.indices = tf.range(self.class_labels.shape[0])
        self.full_set = None

    def encode_labels(self, img):
        """
        params: encoded image and string label
        return: image and one-hot-encoded label
        """
        encoded_label = tf.reshape(tf.where(self.class_labels==self.label), (1,))[0]
        encoded_label = tf.one_hot(self.indices, self.indices.shape[0])[encoded_label]
        faw = encoded_label[0]
        zinc = encoded_label[1]
        nlb = encoded_label[2]
        # encoded_label = tf.convert_to_tensor(encoded_label, dtype=tf.int64)
        return img, (faw, zinc, nlb)

class MultiTaskImageGen2(ImageGenerator):
    """
    create a dataset from a tfrecord file
    """
    def __init__(self, file_path, feature_dict, *, for_train=False, for_cnn=True):
        self.raw_set = tf.data.TFRecordDataset(file_path, num_parallel_reads=8)
        self.feat_dict = feature_dict
        self.count = 0
        # for item in self.raw_set:
        #     self.count += 1
        self.full_set=None
        self.for_cnn = for_cnn
        self.for_train = for_train

    def encode_labels(self, img):
        pass

    def get_num_images(self):
        return self.count

    def parse_dataset(self, example):
        return tf.io.parse_single_example(example, self.feat_dict)

    def extract_fn(self, sample):
        image = decode_image(sample['image'][0])
        labels = tf.sparse.to_dense(sample['labels'])
        # labels = [labels[0]], [labels[1]], [labels[2]]
        labels = [labels[0]], [labels[1]]
        return image, labels

    def split_dataset(self):
        """
        Get training and validation sets: splits 80-20 twice
        """
        self.full_set = self.raw_set\
            .map(self.parse_dataset, num_parallel_calls=AUTOTUNE)\
            .map(self.extract_fn, num_parallel_calls=AUTOTUNE)\
            .shuffle(self.get_num_images())
        test_set = self.full_set.take(int(0.2*self.get_num_images()))
        self.full_set = self.full_set.skip(int(0.2*self.get_num_images()))
        val_set = self.full_set.take(int(0.2*0.8*self.get_num_images()))
        self.full_set = self.full_set.skip(int(0.2*0.8*self.get_num_images()))
        return test_set, val_set

    def get_all(self):
        if not self.for_train:
          return self.raw_set\
              .map(self.parse_dataset, num_parallel_calls=AUTOTUNE)\
              .map(self.extract_fn, num_parallel_calls=AUTOTUNE)
        else:
          return self.raw_set\
              .map(self.parse_dataset, num_parallel_calls=AUTOTUNE)\
              .map(self.extract_fn, num_parallel_calls=AUTOTUNE)
