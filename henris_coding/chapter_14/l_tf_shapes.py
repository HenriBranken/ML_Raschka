import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    arr = np.array([[1., 2., 3., 3.5],
                    [4., 5., 6., 6.5],
                    [7., 8., 9., 9.5]])
    T1 = tf.constant(arr, name="T1")
    print(T1)
    s = T1.get_shape()
    print("The shape of T1 is {}.".format(s))

    T2 = tf.Variable(tf.random_normal(shape=s))
    print(T2)

    T3 = tf.Variable(tf.random_normal(shape=(s.as_list()[0], )))
    print(T3)
