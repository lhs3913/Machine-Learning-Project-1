# Authors: Matthew Ayala, Maanav Contractor, Alvin Liu, Luke Sims
#
# Model: Logistic Regression
# Dataset: Predict if the client will subscribe to a term deposit
# Training Dataset File: '/data/training.csv'
# Testing Dataset File: '/data/test-1.csv' and '/data/test-2.csv'
#
# Dataset Citation: 
# [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
# In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.
#
# Available at: [pdf] http://hdl.handle.net/1822/14838
#               [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import autograd

training_df = pd.read_csv("data/training.csv", sep = ";")
testing_df = pd.read_csv("data/test-1.csv", sep = ";")

features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 
            'month', 'campaign', 'pdays', 'previous', 'poutcome']

# Converts the categorical columns into bianry columns thorugh one-hot encoding
training_df = pd.get_dummies(training_df, columns = ["job", "marital", "education","contact", "day", "month", "poutcome"])


# Converts the binary yes/no columns into 0/1
binary_col = ["default", "housing", "loan", "y"]
training_df[binary_col] = training_df[binary_col].map(lambda x : 1 if x == "yes" else 0)

# Seperate the features into x and the labels into y
x = training_df.drop("y", axis = 1).values
y = training_df["y"].values
print(x[:5])
print(y[:5])

def sigmoid(z):
    """
    Parameters:
        z(float): product of linear combination, XW + b

    Return: 
        returns a probablility between 0 and 1
    """
    return 1 / (1 + np.exp(-z))

def compute_loss(y, predicted_y, weights, reg_strength):
    """
    Compute the distance from predicted outcome to real outcome

    Parameters:
        y(array): actual y for sample
    
        predicted_y(array): predicted probability from the sigmoid function

        weights(array): model weights

        reg_strength(float)(0): If 0, no regularization. If >0, adds L2 penalty, preventing overly large weights

    Return:
        loss(float): avg distance between y's across all samples
    """
    m = y.shape[0] # number of samples
    eps = 1e-9 # avoid log(0)

    #binary cross-entropy, ie: distance between y and predicted_y
    loss = -(1/m) * np.sum(
        y * np.log(predicted_y + eps) + (1 - y) * np.log(1 - predicted_y + eps)
    )

    # L2 regularization penalty
    loss += (reg_strength / (2 * m)) * np.sum(weights * weights)

    return loss

def training_logreg(x, y, lrate, iterations, reg_strength):
    """
    Train logistic regression using gradient descent

    Parameters:
        x(array): feature matrix

        y(array): output matrix

        lrate(float)(0.01): learning rate or step size for gradient descent

        iterations(int)(1000): number of training iterations

        reg_strength(float)(0): L2 regularization strength

    Returns:
        W(array): Learned weights

        b(float): learned bias term

        losses(list): loss value at each iteration
    """
    m, n = x.shape
    W = np.zeros((n, 1)) # initialize weights array
    b = 0 # intialize bias
    losses = []

    for i in range(iterations):
        # forwad pass
        z = np.dot(x, W) + b
        predicted_y = sigmoid(z)

        # gradient for weight(how much each weight should change)
        gradient_weights = (1 / m) * (x.T @ (predicted_y - y)) + (reg_strength / m)

        # gradient for bias
        gradient_bias = (1 / m) * np.sum(predicted_y - y)

        # update step
        W -= lrate * gradient_weights
        b -= lrate * gradient_bias

        # track loss
        losses.append(compute_loss(y, predicted_y, W, reg_strength))

    return W, b, losses

def predict(x, W, b, threshold):
    """
    Predicts the class label for a given input
    
    Parameters:
        x(array): feature array

        W(array): weights

        b(float): bias

        threshold(float)(0.5): cutoff for probability

    Return:
        predicted y value for the feature array
    """
    z = np.dot(x, W) + b
    return (sigmoid(z) >= threshold).astype(int)

