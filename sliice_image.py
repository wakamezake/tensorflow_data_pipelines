"""
Reference:
https://www.tensorflow.org/guide/data
"""

import tensorflow as tf
import tensorflow.keras as keras


def fashion_mnist_generator():
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    image_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    label_dataset = tf.data.Dataset.from_tensor_slices(y_train)
    zipped = tf.data.Dataset.zip((image_dataset, label_dataset))
    for image, label in zipped:
        yield image, label


def main():
    print(tf.__version__)
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    print(x_train.shape, y_train.shape)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    print(dataset)

    batch_size = 32
    g = tf.data.Dataset.from_generator(fashion_mnist_generator,
                                       (tf.uint8, tf.uint8))
    for image, label in g.batch(batch_size=batch_size):
        print(image.shape)
        print(label.shape)
        break


if __name__ == '__main__':
    main()
