import os
import struct
import numpy as np
import tensorflow as tf


def load_mnist(path, kind="train"):
    """Load MNIST data from `path`."""
    labels_path = os.path.join(path, "{:s}-labels-idx1-ubyte".format(kind))
    images_path = os.path.join(path, "{:s}-images-idx3-ubyte".format(kind))

    with open(labels_path, "rb") as lbpath:
        _, _ = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        _, _, _, _ = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


# Before we start training the network, we need a way to generate batches of
# data.  For this, we implement the following function that returns a
# generator.
def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.copy(X)
    y_copy = np.copy(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i: i + batch_size, :], y_copy[i: i + batch_size])


# Loading the data
X_train, y_train = load_mnist("./mnist/", kind="train")
print("Rows: {:.0f}, Columns: {:.0f}.".format(X_train.shape[0],
                                              X_train.shape[1]))
X_test, y_test = load_mnist("./mnist/", kind="t10k")
print("Rows: {:.0f}, Columns: {:.0f}.".format(X_test.shape[0],
                                              X_test.shape[1]))

# Mean centering and normalisation:
mean_val = np.mean(X_train)
std_val = np.std(X_train)
X_train_std = (X_train - mean_val) / std_val
X_test_std = (X_test - mean_val) / std_val

del X_train, X_test

print("X_train_std.shape: {}, y_train.shape: {}.".format(X_train_std.shape,
                                                         y_train.shape))
print("X_test_std.shape: {}, y_test.shape: {}.".format(X_test_std.shape,
                                                       y_test.shape))

n_features = X_train_std.shape[1]  # the number of columns = 784
n_classes = len(np.unique(y_train))
random_seed = 123
np.random.seed(random_seed)

# We start building our model.  We will start by creating two placeholders
# named tf_x and tf_y, and then build a multilayer perceptron, but with 3
# fully connected layers.
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32,
                          shape=(None, n_features),
                          name="tf_x")
    tf_y = tf.placeholder(dtype=tf.int32,
                          shape=None,
                          name="tf_y")
    y_onehot = tf.one_hot(indices=tf_y,
                          depth=n_classes)
    h1 = tf.layers.dense(inputs=tf_x,
                         units=50,
                         activation=tf.tanh,
                         name="layer_1")
    h2 = tf.layers.dense(inputs=h1,
                         units=50,
                         activation=tf.tanh,
                         name="layer_2")
    logits = tf.layers.dense(inputs=h2,
                             units=10,
                             activation=None,
                             name="layer_3")
    predictions = {
        "classes": tf.argmax(logits, axis=1, name="predicted_classes"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Next, define the cost functions, and add an operator for initializing the
    # model variables as well as an optimization operator:
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot,
                                           logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=cost)
    init_op = tf.global_variables_initializer()

# Next, we can create a new TensorFlow session, initialize all the variables in
# our network, and train it.  We also display the average training loss after
# each epoch to monitor the learning process:

# Create a session to launch the graph.
sess = tf.Session(graph=g)
# run the variable initialisation operator.
sess.run(init_op)

# Execute 50 epochs of training:
for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(X=X_train_std, y=y_train,
                                             batch_size=64)
    for batch_X, batch_y in batch_generator:
        # prepare a dictionary to feed our data to our network:
        feed = {tf_x: batch_X, tf_y: batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(" -- Epoch {:2.0f}:\n"
          "Avg. Training Loss: {:.4f}".format(epoch + 1,
                                              np.mean(training_costs)))

# Finally, we can use the trained model to do predictions on the test dataset:
feed = {tf_x: X_test_std}
y_pred = sess.run(predictions['classes'], feed_dict=feed)

print("Test Accuracy: {:.4f}%.".format(
    np.sum(y_pred == y_test) / y_test.shape[0] * 100.0))
