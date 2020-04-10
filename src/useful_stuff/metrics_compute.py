"""
Elementary functions to compute metrics
"""

import tensorflow as tf


def compute_precision(class_label: int, confusion_matrix: tf.Tensor) -> int:
    """
    Compute precision for a single class.
    """
    index = class_label
    def true_func():
        return tf.divide(confusion_matrix[index, index], tf.reduce_sum(confusion_matrix[:, index]))
    def false_func():
        return tf.reduce_sum(confusion_matrix[:, index])
    denominator = tf.reduce_sum(confusion_matrix[:,index])
    return tf.cond(denominator>0, true_func, false_func)

# @tf.function
def compute_recall(class_label: int, confusion_matrix: tf.Tensor) -> int:
    """
    Compute recall for a single class
    """
    index = class_label
    def true_func():
        return tf.divide(confusion_matrix[index, index], tf.reduce_sum(confusion_matrix[index, :]))
    def false_func():
        return tf.reduce_sum(confusion_matrix[index, :])
    denominator = tf.reduce_sum(confusion_matrix[index, :])
    return tf.cond(denominator>0, true_func, false_func)
    
# @tf.function
def compute_f1(index: int, confusion_matrix: tf.Tensor) -> int:
    precision = compute_precision(index, confusion_matrix)
    recall = compute_recall(index, confusion_matrix)
    def true_func():
        return tf.divide(2*tf.multiply(precision,recall), tf.add(precision, recall))
    def false_func():
        return tf.add(precision, recall)
    return tf.cond(precision+recall>0, true_func, false_func)