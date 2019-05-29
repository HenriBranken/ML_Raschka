import tensorflow as tf
import numpy as np
np.random.seed(123)
np.set_printoptions(precision=2)

g = tf.Graph()

with g.as_default():
    # We can specify `None` for the dimension that is varying in size.  E.g.,
    # we can create a placeholder of rank 2, where the first dimension is
    # unknown (or may vary), as shown here:
    tf_x = tf.placeholder(tf.float32, shape=[None, 2], name="tf_x")

    x_mean = tf.reduce_mean(tf_x, axis=0, name="mean")

with tf.Session(graph=g) as sess:
    # We can evaluate `x_mean` with 2 different inputs, x1 and x2:
    x1 = np.random.uniform(low=0, high=1, size=(5, 2))
    print("Feeding data with the shape {}.".format(x1.shape))
    print("Result: {}.".format(sess.run(x_mean, feed_dict={tf_x: x1})))

    x2 = np.random.uniform(low=0, high=1, size=(10, 2))
    print("Feeding data with the shape {}.".format(x2.shape))
    print("Result: {}.".format(sess.run(x_mean, feed_dict={tf_x: x2})))

    # Print the object tf_x:
    print(tf_x)
