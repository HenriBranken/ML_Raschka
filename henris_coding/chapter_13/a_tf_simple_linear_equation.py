import tensorflow as tf

# create a graph
g = tf.Graph()

with g.as_default():
    x = tf.placeholder(dtype=tf.float32,
                       shape=None,
                       name="x")
    w = tf.Variable(2.0, name="weight")
    b = tf.Variable(0.7, name="bias")

    z = w*x + b

    init = tf.global_variables_initializer()

# Create a Session and pass in graph g
with tf.Session(graph=g) as sess:
    # initialize w and b
    sess.run(init)
    # Evaluate z
    # Feed in the values in an element-by-element form.
    for x_val in [1.0, 0.6, -1.8]:
        print("x={:4.1f} --> z={:4.1f}".format(x_val,
                                               sess.run(z,
                                                        feed_dict={x: x_val})))
    # Feed in the values as a batch of input data at once.
    print(sess.run(z, feed_dict={x: [1., 2., 3.]}))
