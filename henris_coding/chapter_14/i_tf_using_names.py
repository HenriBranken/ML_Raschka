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


x, y = make_random_data()
# plt.plot(x, y, "o")
# plt.show()
# plt.clf()
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]
n_epochs = 500
training_costs = []

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(123)

    # Placeholders
    tf_x = tf.placeholder(shape=None, dtype=tf.float32, name="tf_x")
    tf_y = tf.placeholder(shape=None, dtype=tf.float32, name="tf_y")

    # Define the variables/model parameters
    weight = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.25),
                         name="weight")
    bias = tf.Variable(0.0, name="bias")

    # Build the model
    y_hat = tf.add(weight * tf_x, bias, name="y_hat")

    # Compute the cost
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name="cost")

    # Train the model
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    training_op = optim.minimize(cost, name="training_op")

# Now let us train the model:
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    # train the model for n_epochs
    for e in range(n_epochs + 1):
        c, _ = sess.run(["cost:0", "training_op"],
                        feed_dict={"tf_x:0": x_train, "tf_y:0": y_train})
        training_costs.append(c)
        if not e % 50:
            print("Epoch {:>4d}: {:.4f}.".format(e, c))

    plt.plot(training_costs)
    plt.show()
    print("weight = {}, bias = {}.".format(weight.eval(), bias.eval()))
