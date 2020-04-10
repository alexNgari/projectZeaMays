"""
Custom metrics for evaluating models
"""

import numpy as np
import tensorflow as tf
from src.useful_stuff.metrics_compute import compute_f1

# class ConfusionMatrix(tf.keras.metrics.Metric):

#     def __init__(self, num_classes, name='conf_mat', **kwargs):
#         super(ConfusionMatrix, self).__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.conf_mat = self.add_weight(name='conf_mat', shape=(num_classes, num_classes), initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.argmax(y_true, axis=1)
#         y_true = tf.cast(y_true, tf.int32)
#         y_pred = tf.argmax(y_pred, axis=1)
#         y_pred = tf.cast(y_pred, tf.int32)
#         conf_mat = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
#         if sample_weight:
#           sample_weight = tf.cast(sample_weight, self.dtype)
#           sample_weight = tf.broadcast_weights(sample_weight, conf_mat)
#           conf_mat = tf.multiply(conf_mat, sample_weight)
#         self.conf_mat.assign_add(conf_mat)

#     def result(self):
#         return self.conf_mat.numpy()

#     def reset_states(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.conf_mat.assign(tf.zeros(self.conf_mat.shape))


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int, subject_class: int=None, name: str='f1', **kwargs):
        """
        Args:
            num_classes:    number of classes in multiclass problem
            subject_class:  the class to compute score for
            name:           name for the metric in logs
        Attributes:
            num_classes
            subject_class
            conf_mat:       confusion matrix, which is updated every epoch. Required for computation of score
        """
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.subject_class = subject_class
        self.conf_mat = self.add_weight(name='conf_mat', shape=(num_classes, num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update confusion matrix after epoch"""
        y_true = tf.argmax(y_true, axis=1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=1)
        y_pred = tf.cast(y_pred, tf.int32)
        conf_mat = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        if sample_weight:
          sample_weight = tf.cast(sample_weight, self.dtype)
          sample_weight = tf.broadcast_weights(sample_weight, conf_mat)
          conf_mat = tf.multiply(conf_mat, sample_weight)
        self.conf_mat.assign_add(conf_mat)

    def result(self):
        """
        Returns:
            f1 score for a specific class if subject_class was passed in __init__,
            or weighted average f1 score if subject_class was left None
        """
        try:
            a = tf.convert_to_tensor(self.subject_class)
            return compute_f1(self.subject_class, self.conf_mat)
        except ValueError:
            f1 = 0
            weights = tf.reduce_sum(self.conf_mat, axis=1)
            for i in range(self.num_classes):
                f1 += weights[i] * compute_f1(i, self.conf_mat)
            return f1/tf.reduce_sum(weights)



    def reset_states(self):
        """Reset confusion matrix after epoch"""
        # The state of the metric will be reset at the start of each epoch.
        self.conf_mat.assign(tf.zeros(self.conf_mat.shape))