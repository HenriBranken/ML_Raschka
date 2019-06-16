import numpy as np
import os
import struct
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


# Loading the data:
X_data, y_data = load_mnist("./mnist/", kind="train")
print("Data Rows:   {}, Columns: {}.".format(X_data.shape[0], X_data.shape[1]))
X_test, y_test = load_mnist("./mnist/", kind="t10k")
print("Test Rows:   {}, Columns: {}.".format(X_test.shape[0], X_test.shape[1]))

X_train, y_train = X_data[: 50000, :], y_data[: 50000]
X_valid, y_valid = X_data[50000:, :], y_data[50000:]

print("Training:    {}, {}.".format(X_train.shape, y_train.shape))
print("Validation:  {}, {}.".format(X_valid.shape, y_valid.shape))
print("Test set:    {}, {}.".format(X_test.shape, y_test.shape))

# Normalise the data:
mean_val = np.mean(X_train)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_val)/std_val
X_valid_centered = (X_valid - mean_val)/std_val
X_test_centered = (X_test - mean_val)/std_val


def batch_generator(X, y, batch_size=64, shuffle=False, random_seed=None):
    """A function for iterating through mini-batches of the data."""

    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        # Container for the Mersenne Twister pseudo-random number generator.
        rng.shuffle(idx)  # Modify a sequence INPLACE by shuffling its contents
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield X[i: i + batch_size, :], y[i: i + batch_size]


def conv_layer(input_tensor, name, kernel_size, n_output_channels,
               padding_mode="SAME", strides=(1, 1, 1, 1)):
    """A Wrapper function for building a Convolutional Layer, including:
        (1) defining the weights
        (2) defining the biases
        (3) initializing the weights and biases
        (4) performing convolution operation."""
    with tf.variable_scope(name):
        # get the number of input channels, `n_input_channels`.
        # input tensor shape is [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = list(kernel_size) + \
                        [n_input_channels, n_output_channels]

        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print("Weights --> {}.".format(weights))

        biases = \
            tf.get_variable(name="_biases",
                            initializer=tf.zeros(shape=[n_output_channels]))
        print("Biases --> {}.".format(biases))

        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding_mode)
        print(conv)

        conv = tf.nn.bias_add(conv, biases, name="net_pre-activation")
        print(conv)

        conv = tf.nn.relu(conv, name="activation")  # Rectified linear.
        print(conv)

        return conv


# Let us test this function with a simple input by defining a placeholder, as
# follows:
# g = tf.Graph()
# with g.as_default():
#     x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
#     conv_layer(x, name="convtext", kernel_size=(3, 3), n_output_channels=32)
# del g, x

# The next wrapper function is for defining our fully connected layers:
def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    """Defining our fully connected layers, including:
        (1) building weights and biases,
        (2) initializing weights and biases,
        (3) performs matrix multiplication using the tf.matmul function"""
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        # Return the product of array elements over a given axis.
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print(weights)

        biases = tf.get_variable(name="_biases",
                                 initializer=tf.zeros(shape=[n_output_units]))
        print(biases)

        layer = tf.matmul(input_tensor, weights)
        print(layer)

        layer = tf.nn.bias_add(layer, biases, name="net_pre-activation")
        print(layer)

        if activation_fn is None:
            return layer

        layer = activation_fn(layer, name="activation")
        print(layer)

        return layer


# We can test the fc_layer function for a simple input tensor as follows:
# g = tf.Graph()
# with g.as_default():
#     x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
#     fc_layer(x, name="fctest", n_output_units=32, activation_fn=tf.nn.relu)
#
# del g, x

def build_cnn():
    """Build the whole convolutional network.  build_cnn handles the building
    of the CNN model, as shown in the following code."""
    # Define placehlder for x and y.
    tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="tf_x")
    tf_y = tf.placeholder(dtype=tf.int32, shape=[None], name="tf_y")

    # Reshape x to a 4D tensor -> [batchsize, width, height, 1]:
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name="tf_x_reshaped")

    # One-Hot Encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32,
                             name="tf_y_onehot")

    # 1st Layer:  conv_1
    print("\nBuilding the 1st layer:")
    h1 = conv_layer(input_tensor=tf_x_image, name="conv_1", kernel_size=(5, 5),
                    n_output_channels=32, padding_mode="VALID")

    # Max Pooling
    h1_pool = tf.nn.max_pool(value=h1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding="SAME")

    # 2nd Layer:  conv_2
    print("\nBuilding the 2nd layer:")
    h2 = conv_layer(input_tensor=h1_pool, name="conv_2", kernel_size=(5, 5),
                    n_output_channels=64, padding_mode="VALID")

    # Max Pooling
    h2_pool = tf.nn.max_pool(value=h2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding="SAME")

    # 3rd Layer:  Fully Connected (fc).
    print("\nBuilding the 3rd layer:")
    h3 = fc_layer(input_tensor=h2_pool, name="fc_3", n_output_units=1024,
                  activation_fn=tf.nn.relu)

    # Dropout
    # For each element of x, with probability rate, outputs 0, and otherwise
    # scales up the input by 1 / (1 - rate).  The scaling is such that the
    # expected sum is unchanged.
    keep_prob = tf.placeholder(tf.float32, name="fc_keep_prob")
    h3_drop = tf.nn.dropout(x=h3, rate=1.0 - keep_prob, name="dropout_layer")

    # 4th layer:  Fully Connected (Linear Activation)
    print("Building the 4th layer:")
    h4 = fc_layer(input_tensor=h3_drop, name="fc_4", n_output_units=10,
                  activation_fn=None)

    # Prediction:
    predictions = {
        "probabilities": tf.nn.softmax(h4, name="probabilities"),
        "labels": tf.cast(x=tf.argmax(h4, axis=1), dtype=tf.int32,
                          name="labels")
    }

    # Visualise the graph with TensorBoard.