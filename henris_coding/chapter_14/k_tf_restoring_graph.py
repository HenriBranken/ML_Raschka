import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    noise = []
    for _ in x:
        r = np.random.normal(loc=0.0, scale=1.0, size=None)
        noise.append(r)
    return x, 1.726*x - 0.84 + np.array(noise)


x_arr = np.arange(-2, 4, 0.1)
x, y = make_random_data()
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]
n_epochs = 500
training_costs = []

g = tf.Graph()

# Now let us train the model:
with tf.Session(graph=g) as sess:
    new_saver = tf.train.import_meta_graph("./trained-model.meta")

    new_saver.restore(sess, "./trained-model")

    y_line = sess.run("y_hat:0",
                      feed_dict={"tf_x:0": x_arr})

plt.figure()
plt.plot(x_train, y_train, "bo", alpha=0.5)
plt.plot(x_test, y_test, "ko", alpha=0.5)
plt.plot(x_arr, y_line.T[:, 0], "-r", lw=3)
plt.show()
