"""
Reference:
https://www.tensorflow.org/guide/data
"""
from pprint import pprint

import tensorflow as tf


def main():
    print(tf.__version__)
    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
    print(dataset)

    # generator?
    print("once")
    pprint([(data, data.numpy()) for data in dataset])

    print("twice")
    pprint([(data, data.numpy()) for data in dataset])


if __name__ == '__main__':
    main()
