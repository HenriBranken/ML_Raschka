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

    T5 = tf.reshape(T1, shape=[1, -1, 3], name="T5")
    print(T5)

    t5_splt = tf.split(T5, num_or_size_splits=2, axis=1, name="T8")
    print(t5_splt)
with tf.Session(graph=g) as sess:
    print(sess.run(T5))
    print(sess.run([t5_splt[0], t5_splt[1]]))
