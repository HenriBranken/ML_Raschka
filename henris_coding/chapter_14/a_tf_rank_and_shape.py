import tensorflow as tf


g = tf.Graph()

# Define the computation graph
with g.as_default():
    # Define the tensors t1, t2, and t3.
    t1 = tf.constant(1)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])

    # Get the ranks of the tensors.  It returns a tensor as output, and we will
    # need to evaluate the tensor to get the actual value.
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)

    # Get the shapes of the tensors.  It returns an object of a special class
    # called TensorShape.
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    print("Shapes: ", s1, s2, s3)

with tf.Session(graph=g) as sess:
    print("Ranks: ", r1.eval(), r2.eval(), r3.eval())
