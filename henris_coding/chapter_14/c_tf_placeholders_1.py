import tensorflow as tf


g = tf.Graph()

with g.as_default():
    # Use placeholders for the scalars a, b, and c:
    tf_a = tf.placeholder(tf.int32, shape=[], name="tf_a")
    tf_b = tf.placeholder(tf.int32, shape=[], name="tf_b")
    tf_c = tf.placeholder(tf.int32, shape=[], name="tf_c")

    # Store the intermediate tensors associated with r1 and r2 as follows:
    r1 = tf_a - tf_b
    r2 = 2 * r1

    # The final, 'result' tensor:
    z = r2 + tf_c

# Evaluate the result tensor z:
with tf.Session(graph=g) as sess:
    # This dictionary is passed as the input argument `feed_dict` to a
    # session's `run` method.
    feed = {tf_a: 1,
            tf_b: 2,
            tf_c: 3}
    print("z = {:.0f}.".format(sess.run(z, feed_dict=feed)))
