import tensorflow as tf


# A graph can be created by calling tf.Graph()
g = tf.Graph()

# Nodes can be added to it as follows.
with g.as_default():
    a = tf.constant(1, name="a")
    b = tf.constant(2, name="b")
    c = tf.constant(3, name="c")

    z = 2 * (a - b) + c

# Launch the previous graph in a tf Session and evaluate the tensor z as
# follows:
with tf.Session(graph=g) as sess:
    print("2 * (a - b) + c = {}.".format(sess.run(z)))
