import os
import struct
import numpy as np
import tensorflow as tf
import keras


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

n_features = X_train_std.shape[1]  # the number of columns = 784
n_classes = len(np.unique(y_train))

# First, let's set the random seed for Numpy and TensorFlow so that we get
# consistent results.
random_seed = 123
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

# We need to convert the class labels (integers 0-9) into the one-hot format.
y_train_onehot = keras.utils.to_categorical(y_train)
print("First 3 labels: {}.".format(y_train[:3]))
print("\nFirst 3 one-hot labels:\n{}.".format(y_train_onehot[:3]))

# Implement 3 dense layers.  The first 2 will have 50 hidden units each with
# the tanh activation function and the last layer has 10 layers for the 10
# class labels.
# The last layer uses `softmax` to give the probability of each class.
model = keras.models.Sequential()

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=X_train_std.shape[1],
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        activation="tanh"
    )
)

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=50,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        activation="tanh"
    )
)

model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=50,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        activation="softmax"
    )
)

sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)

model.compile(optimizer=sgd_optimizer, loss="categorical_crossentropy")

# Now we can train the model by calling the fit method.
# We use mini-batch SGD with a batch size of 64 training samples per batch.
# We train the MLP over 50 epochs, and we can follow the optimization of the
# cost function during training by setting verbose=1.
# validation_split parameter is handy for validation after each epoch so that
# we can monitor whether the model is overfitting during the training.
history = model.fit(X_train_std, y_train_onehot, batch_size=64,
                    epochs=50, verbose=1, validation_split=0.1)

# To predict the class labels, we can use the `predict_classes` method to
# return the class labels directly as integers.
# We will also print the model accuracy on the training and test sets.
y_train_pred = model.predict_classes(X_train_std, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print("Training accuracy: {:.2f}%.".format(train_acc * 100))

y_test_pred = model.predict_classes(X_test_std, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print("Testing accuracy: {:.2f}%.".format(test_acc * 100))
