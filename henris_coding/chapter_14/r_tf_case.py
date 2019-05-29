import tensorflow as tf

g = tf.Graph()

with g.as_default():
    tf_x = tf.placeholder(tf.float32, shape=None, name="tf_x")
    tf_y = tf.placeholder(tf.float32, shape=None, name="tf_y")

    f1 = lambda: tf.constant(17)
    f2 = lambda: tf.constant(23)

    result = tf.case([(tf.less(tf_x, tf_y), f1)], default=f2)
    print(result)

with tf.Session(graph=g) as sess:
    x, y = 1.0, 2.0
    print("x < y: {} "
          "-> result = {}.".format(x < y,
                                   sess.run(result,
                                            feed_dict={"tf_x:0": x,
                                                       "tf_y:0": y})))
    x, y = 2.0, 1.0
    print("x < y: {} "
          "-> result = {}.".format(x < y,
                                   result.eval(feed_dict={"tf_x:0": x,
                                                          "tf_y:0": y})))
