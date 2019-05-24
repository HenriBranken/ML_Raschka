import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a small one-dimensional toy dataset with 10 training samples.
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])


class TfLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()

        # build the model
        with self.g.as_default():
            # set the graph-level random seed
            tf.set_random_seed(random_seed)

            self.build()

            # create initializer
            self.init_op = tf.global_variables_initializer()

    def build(self):
        # define the placeholders for the inputs
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=(None, self.x_dim),
                                name="x_input")
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=None,
                                name="y_input")
        print(self.X)
        print(self.y)

        # define the weight matrix and the bias vector
        w = tf.Variable(tf.zeros(shape=1),
                        name="weight")
        b = tf.Variable(tf.zeros(shape=1),
                        name="bias")
        print(w)
        print(b)

        self.z_net = tf.squeeze(w*self.X + b,
                                name="z_net")
        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net,
                               name="sqr_errors")
        print(sqr_errors)

        self.mean_cost = tf.reduce_mean(sqr_errors,
                                        name="mean_cost")

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
            name="GradientDescent")
        self.optimizer = optimizer.minimize(self.mean_cost)


# Next we implement a training function to learn the weights of the linear
# regression model.  Note that b is the bias unit.
# For training, we implement a separate function that needs a TensorFlow
# session, a model instance, training data, and the number of epochs as input
# arguments.
def train_linreg(sess, model, X_tr, y_tr, num_epochs=10):
    # initialize all the variables: w and b
    sess.run(model.init_op)

    tr_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost],
                           feed_dict={model.X: X_tr,
                                      model.y: y_tr})
        tr_costs.append(cost)
    return tr_costs


# Compile a new function to make predictions based on the input features.
def predict_linreg(sess, model, X_test):
    y_pred = sess.run(model.z_net,
                      feed_dict={model.X: X_test})
    return y_pred


# Create an instance of the class and call it `lrmodel` as follows
lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)
sess = tf.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)

# Let us visualize the training costs after these 10 epochs to see whether the
# model has converged or not:
# plt.plot(range(1, len(training_costs) + 1), training_costs)
# plt.tight_layout()
# plt.xlabel("Epoch")
# plt.ylabel("Training Cost")
# plt.show()

# Next, let us plot the linear regression fit on the training data.
plt.scatter(X_train, y_train, marker="s", s=50, label="Training data")
plt.plot(X_train, predict_linreg(sess, lrmodel, X_train), color="gray",
         marker="o", markersize=6, lw=3, label="LinReg Model")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

# As we can see in the resulting plot, our model fits the training data points
# appropriately.
