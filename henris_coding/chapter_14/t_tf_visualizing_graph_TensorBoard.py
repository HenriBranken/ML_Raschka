import tensorflow as tf


def build_classifier(data, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name="weights", shape=(data_shape[1], n_classes),
                              dtype=tf.float32)
    bias = tf.get_variable(name="bias", initializer=tf.zeros(shape=n_classes))
    logits = tf.add(tf.matmul(data, weights), bias, name="logits")
    return logits, tf.nn.softmax(logits)


def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(tf.random_normal(shape=(data_shape[1], n_hidden)),
                     name="w1")
    b1 = tf.Variable(tf.zeros(shape=n_hidden), name="b1")
    hidden = tf.add(tf.matmul(data, w1), b1, name="hidden_pre-activation")
    hidden = tf.nn.relu(hidden, name="hidden_activation")

    w2 = tf.Variable(tf.random_normal(shape=(n_hidden, data_shape[1])),
                     name="w2")
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]), name="b2")
    output = tf.add(tf.matmul(hidden, w2), b2, name="output")
    return output, tf.nn.sigmoid(output)  # logistic function.


# Build the Graph

batch_size = 64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32,
                          name="tf_X")

    # Build the generator
    with tf.variable_scope("generator"):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)

    # Build the classifier
    with tf.variable_scope("classifier"):
        # Classifier for the original data
        cls_out1 = build_classifier(data=tf_X)

    with tf.variable_scope("classifier", reuse=True):
        # reuse the classifier for generated data.
        cls_out2 = build_classifier(data=gen_out1[1])

# The following lines of code export the graph for visualization purposes:
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    file_writer = tf.summary.FileWriter(logdir="./logs/",
                                        graph=g)
    # This will create a new directory: "./logs/".  Now we just need to run the
    # following command from our terminal:
    # tensorboard --logdir logs/