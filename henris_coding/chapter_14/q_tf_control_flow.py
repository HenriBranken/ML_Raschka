import tensorflow as tf

x, y = 1.0, 2.0

g = tf.Graph()

with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, shape=None, name="tf_x")
    tf_y = tf.placeholder(dtype=tf.float32, shape=None, name="tf_y")

    res = tf.cond(tf_x < tf_y,
                  true_fn=lambda: tf.add(tf_x, tf_y, name="result_add"),
                  false_fn=lambda: tf.subtract(tf_x, tf_y, name="result_sub"))
    print("Object: {}.".format(res))

with tf.Session(graph=g) as sess:
    print("x < y: {} -> "
          "Result: {}.".format(x < y, res.eval(feed_dict={"tf_x:0": x,
                                                          "tf_y:0": y})))

    x, y = 2.0, 1.0
    print("x < y: {} -> "
          "Result: {}.".format(x< y, res.eval(feed_dict={"tf_x:0": x,
                                                         "tf_y:0": y})))
