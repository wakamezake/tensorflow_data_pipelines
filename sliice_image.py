"""
Reference:
https://www.tensorflow.org/guide/data
"""

from functools import partial

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical


def fashion_mnist_generator():
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    image_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    label_dataset = tf.data.Dataset.from_tensor_slices(y_train)
    zipped = tf.data.Dataset.zip((image_dataset, label_dataset))
    for image, label in zipped:
        yield image, label


def main():
    print('tf.executing_eagerly() = ', tf.executing_eagerly())
    print(tf.__version__)
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    print(x_train.shape, y_train.shape)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    print(dataset)

    num_classes = len(set(y_train))
    to_cat = partial(tf.one_hot, depth=num_classes)
    image_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: x / 255)
    label_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(to_cat)
    zipped_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    for image, label in zipped_dataset:
        print(image)
        print(label)
        break

    batch_size = 32
    g = tf.data.Dataset.from_generator(fashion_mnist_generator,
                                       (tf.uint8, tf.uint8))
    for image, label in g.batch(batch_size=batch_size):
        print(image.shape)
        print(label.shape)
        break


if __name__ == '__main__':
    main()
