import tensorflow as tf

g = tf.Graph()

with g.as_default():
    i = tf.constant(0)
    threshold = 100

    def c(idx):
        return tf.less(idx, threshold)


    def b(idx):
        return tf.add(idx, 1)

    r = tf.while_loop(cond=c, body=b, loop_vars=[i])
    print(r)

with tf.Session(graph=g) as sess:
    print(sess.run(r))
