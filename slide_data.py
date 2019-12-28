"""
Reference:
https://www.tensorflow.org/guide/data
"""
from pprint import pprint

import tensorflow as tf


def main():
    print('tf.executing_eagerly() = ', tf.executing_eagerly())
    print(tf.__version__)
    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
    print(dataset)

    # generator?
    print("once")
    pprint([(data, data.numpy()) for data in dataset])
    """
    [(<tf.Tensor: id=8, shape=(), dtype=int32, numpy=8>, 8),
     (<tf.Tensor: id=9, shape=(), dtype=int32, numpy=3>, 3),
     (<tf.Tensor: id=10, shape=(), dtype=int32, numpy=0>, 0),
     (<tf.Tensor: id=11, shape=(), dtype=int32, numpy=8>, 8),
     (<tf.Tensor: id=12, shape=(), dtype=int32, numpy=2>, 2),
     (<tf.Tensor: id=13, shape=(), dtype=int32, numpy=1>, 1)]
    """

    print("twice")
    pprint([(data, data.numpy()) for data in dataset])
    """
    [(<tf.Tensor: id=19, shape=(), dtype=int32, numpy=8>, 8),
     (<tf.Tensor: id=20, shape=(), dtype=int32, numpy=3>, 3),
     (<tf.Tensor: id=21, shape=(), dtype=int32, numpy=0>, 0),
     (<tf.Tensor: id=22, shape=(), dtype=int32, numpy=8>, 8),
     (<tf.Tensor: id=23, shape=(), dtype=int32, numpy=2>, 2),
     (<tf.Tensor: id=24, shape=(), dtype=int32, numpy=1>, 1)]
    """

    batch_size = 2
    batched_dataset = dataset.batch(batch_size)

    print("batch_size: {}".format(batch_size))
    pprint([(data, data.numpy()) for data in batched_dataset])
    """
    batch_size: 2
    [(<tf.Tensor: id=33, shape=(2,), dtype=int32, numpy=array([8, 3])>,
      array([8, 3])),
     (<tf.Tensor: id=34, shape=(2,), dtype=int32, numpy=array([0, 8])>,
      array([0, 8])),
     (<tf.Tensor: id=35, shape=(2,), dtype=int32, numpy=array([2, 1])>,
      array([2, 1]))]
    """

    batch_size = 4
    batched_dataset = dataset.batch(batch_size)

    print("batch_size: {}".format(batch_size))
    pprint([(data, data.numpy()) for data in batched_dataset])
    """
    batch_size: 4
    [(<tf.Tensor: id=44, shape=(4,), dtype=int32, numpy=array([8, 3, 0, 8])>,
      array([8, 3, 0, 8])),
     (<tf.Tensor: id=45, shape=(2,), dtype=int32, numpy=array([2, 1])>,
      array([2, 1]))]
    """


if __name__ == '__main__':
    main()
