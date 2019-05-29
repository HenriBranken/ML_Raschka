import tensorflow as tf


g = tf.Graph()

with g.as_default():
    t1 = tf.ones(shape=(5, 1), dtype=tf.float32, name="t1")
    t2 = tf.zeros(shape=(5, 1), dtype=tf.float32, name="t2")
    print(t1)
    print(t2)

    t3 = tf.concat([t1, t2], axis=0, name="t3")
    print(t3)
    t4 = tf.concat([t1, t2], axis=1, name="t4")
    print(t4)

with tf.Session(graph=g) as sess:
    print(t3.eval())
    print(t4.eval())
