
# coding: utf-8

# # The Perceptron Classifier
# 
# ## Brief Theory Background
# 
# Rosenblatt's perceptron rule can be summarized by the following steps:
# 1.  Initialise the weights to 0 or small random numbers.
# 2.  For each training sample $\textbf{x}^{(i)}$:
#     1.  Compute the output value $\hat{y}$.
#     2.  Update the weights.
# 
# The simulataneous update of each weight $w_j$ in the weight vector $\textbf{w}$ can be more formally written as:
# $$
# w_j := w_j + \Delta w_j
# $$
# 
# The value of $\Delta w_j$, which is used to update the weight $w_j$, is calculated by the perceptron learning rule:
# $$
# \Delta w_j = \eta \left( y^{(i)} - \hat{y}^{(i)} \right) x_{j}^{(i)}
# $$
# 
# Where $\eta$ is the learning rate, $y^{(i)}$ is the true class label of the $i^{\mathrm{th}}$ training sample, and $\hat{y}^{(i)}$ is the predicted class label.

# In[5]:


import numpy as np

class Perceptron(object):
    """
    The perceptron classifier.
    
    ------------------------------
    The parameters are as follows:
    ------------------------------
    eta: float
        The Learning Rate (between 0.0 and 1.0)
    
    n_iter:  int
        The number of passes over the training dataset.
    
    random_state: int
        Random number generator seed for random weight initialization.
    
    ------------------------------
    The attributes are as follows:
    ------------------------------
    w_:  1d-array
        Weights after fitting.
    
    errors_:  list
        Number of misclassifications (updates) in each epoch
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the training data.
        
        ------------------------------
        The parameters are as follows:
        ------------------------------
        X:  {array_like} with shape:
                n_samples rows by n_features columns.
            Each row in X is a training vector.
            n_samples is the total number of samples (e.g. total number of Iris flowers)
            n_features is the total number of features per sample.

        y:  {array_like} with shape = [n_samples]
            y represents the target values, i.e., the true class labels.

        -----------------------------
        The return of this method is:
        -----------------------------
        self:  object
        """
        
        rgen = np.random.RandomState(self.random_state)
        # RandomState exposes a number of methods for generating random numbers drawn from a
        # variety of probability distributions.
        # In addition to the distribution-specific arguments, each method takes a keyword argument
        # size that defaults to None.  
        # If size is None, then a single value is generated and returned.
        # If size is an integer, then a 1-D array filled with generated values is returned
        
        # self.w_ = np.zeros(1 + X.shape[1])
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # normal([loc, scale, size]) 
        # Draw random samples from a normal (Gaussian) distribution.
        # X.shape[1] represents the total number of columns / features per sample.
        
        self.errors_ = []  # self.errors_ is an empty list.
        
        for _ in range(self.n_iter):
            # iterate over the total number of passes over the training dataset.
            # n_iter defaults to 50.
            errors = 0
            for xi, target in zip(X, y):  # iterate over the ith sample with corresponding true target label y.
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi  # update the weights
                self.w_[0] += update * 1.0  # update the bias unit
                errors += int(update != 0)  # where target and self.predict(xi) are not equal to each other.
            self.errors_.append(errors)  # append the number of misclassifications to the growing list.
        return self
    
    def predict(self, X):
        """
        Return the predicted class label after unit step.
        This is y hat.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        # numpy.where(condition[, x, y])
        # when condition is True, yield +1.  Otherwise (i.e. when condition is False), yield -1.
    
    def net_input(self, X):
        """
        Calculate the net input.
        The net input is a dot product between the weights and the input signal.
        """
        return np.dot(X, self.w_[1:] + self.w_[0])


# # Comments on the above-defined class object
# 
# Via the ```fit``` method, we initialize the weights in ```self.w_``` to a vector ${\rm I\!R}^{m+1}$.
# $m$ stands for the number of dimensions (features) in the dataset, where we add 1 for the first element in this vector that represents the bias unit.  Remember that the first element in this vectory, ```self.w_[0]```, represents the so-called bias units.
# 
# Also notice that this vector contains small random numbers drawn from a normal distribution with standard deviation ```0.01``` via ```rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])```, where ```rgen``` is a NumPy random number generator that we seeded with a user-specified random seed so that we can reproduce the previous results if so desired.
# 
# Now, the reason we don't initialise the weights to zero, is that the learning rate $\eta$ only has an effect on the classification outcome if the weights are initialized to non-zero values.  If all the weights are initalized to zero, the learning rate parameter affects only the scale of the weight vector, no the direction.
# 
# After the weights have been initialised, the ```fit``` method loops over all the individual samples in the training set and updates the weights according to the perceptron learning rule that we discussed in the previous section.  The class labels are predicted by the predict method.
# 
# Furthermore, we also collect the number of misclassifications during each epoch in the ```self.errors_``` list so that we can later analyze how well our perceptron performed during the training.
