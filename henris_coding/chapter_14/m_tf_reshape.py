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

    T4 = tf.reshape(T1, shape=[1, 1, -1], name="T4")
    print(T4)

    T5 = tf.reshape(T1, shape=[1, -1, 3], name="T5")
    print(T5)

with tf.Session(graph=g) as sess:
    print(sess.run(T4))
    print(sess.run(T5))
