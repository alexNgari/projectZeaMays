"""
Convert a folder of images into a tfrecord file
"""
#%%
import os
import tensorflow as tf

#%%
class GenerateTFRecord:
    def __init__(self, labels):
        self.class_labels: tf.Tensor = tf.convert_to_tensor(labels)
        self.indices = tf.range(self.class_labels.shape[0])
        self.label = ''

    def convert_image_folder(self, img_folder, tfrecord_file_name):
        # Get all file names of images present in folder
        self.label = img_folder.split('/')[-1]
        encoded_label = self._get_label_with_filename()
        img_paths = tf.io.gfile.listdir(img_folder)
        img_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in img_paths]

        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path in img_paths:
                example = self._convert_image(img_path, encoded_label)
                writer.write(example.SerializeToString())

    def _convert_image(self, img_path, label):
        image_data = tf.io.read_file(img_path)
        img_shape = tf.image.decode_jpeg(image_data, channels=3).shape
        # Read image data in terms of bytes
        # with tf.gfile.FastGFile(img_path, 'rb') as fid:
        #     image_data = fid.read()

        example = tf.train.Example(features = tf.train.Features(feature = {
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[2]])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data.numpy()])),
            'labels': tf.train.Feature(float_list = tf.train.FloatList(value = tf.unstack(label))),
        }))
        return example

    def _get_label_with_filename(self):
        """
        params: encoded image and string label
        return: image and one-hot-encoded label
        """
        encoded_label = tf.reshape(tf.where(self.class_labels==self.label), (1,))[0]
        encoded_label = tf.one_hot(self.indices, self.indices.shape[0])[encoded_label]
        faw = encoded_label[0]
        zinc = encoded_label[1]
        nlb = encoded_label[2]
        return (faw, zinc)

#%%
if __name__ == '__main__':
    DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clean')
    CLASS_LABELS = ['faw', 'zinc_def', 'healthy']
    t = GenerateTFRecord(CLASS_LABELS)
    t.convert_image_folder(os.path.join(DATADIR, 'final/healthy'), os.path.join(DATADIR,'final/healthy.tfrecord'))

# %%
