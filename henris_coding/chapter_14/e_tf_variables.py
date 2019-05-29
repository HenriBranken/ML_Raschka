import tensorflow as tf
import numpy as np


# Variables are a special type of tensor objects that allow us to store and
# update the parameters of our models in a TensorFlow session during training,
# for example: the weights and biases.
g = tf.Graph()

with g.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]]),
                    name="w")
    print(w)

    # Returns an operator for initializing all the variables that exists in the
    # computation graph.
    # Make sure that this operator is created after all the variables have been
    # defined in the computation graph.
    init_op = tf.global_variables_initializer()

with tf.Session(graph=g) as sess:
    # Before executing any node in the computation graph, we must initialize
    # the variables that are within the path to the node that we want to
    # execute.  Executing the tf.global_variables_initializer() operator will
    # initialize the variables.
    init_op.run()
    # Alternatively:  sess.run(init_op)

    # Execute the `w` node defined in the computation graph.
    print(sess.run(w))
