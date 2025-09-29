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
import matplotlib.pyplot as plt
import scipy
import autograd

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

        history(list): history of weights and bias for plotting
    """
    m, n = x.shape
    W = np.zeros((n, 1)) # initialize weights array
    b = 0 # intialize bias
    losses = []
    history = []

    for i in range(iterations):
        # forwad pass
        z = np.dot(x, W) + b
        predicted_y = sigmoid(z)

        # gradient for weight(how much each weight should change)
        gradient_weights = (1 / m) * (x.T @ (predicted_y - y)) + (reg_strength / m) * W

        # gradient for bias
        gradient_bias = (1 / m) * np.sum(predicted_y - y)

        # update step
        W -= lrate * gradient_weights
        b -= lrate * gradient_bias

        # track loss
        losses.append(compute_loss(y, predicted_y, W, reg_strength))
        history.append((W.copy(), b))  # save weights/bias for plotting later

    return W, b, losses, history

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

def evaluate_model(y, predicted_y):
    """
    Compute evaluation metrics for classification.

    Parameters:
        y(array): actual labels (0 or 1)
        predicted_y(array): predicted labels (0 or 1)

    Returns:
        metrics(dict): dictionary with human-readable metrics
    """
    y = y.ravel()
    predicted_y = predicted_y.ravel()

    true_positives = np.sum((predicted_y == 1) & (y == 1))
    true_negatives = np.sum((predicted_y == 0) & (y == 0))
    false_positives = np.sum((predicted_y == 1) & (y == 0))
    false_negatives = np.sum((predicted_y == 0) & (y == 1))

    accuracy = (true_positives + true_negatives) / len(y)
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = 2 * precision * recall / (precision + recall + 1e-9)

    # Print results in a table format
    print("\nEvaluation Report")
    print("-" * 50)
    print(f"{'Overall Accuracy':40}: {accuracy*100:6.2f}%")
    print(f"{'Precision (predicted yes correct)':40}: {precision*100:6.2f}%")
    print(f"{'Recall (actual yes found)':40}: {recall*100:6.2f}%")
    print(f"{'F1 Score (balance of precision/recall)':40}: {f1_score*100:6.2f}%")
    print("-" * 50)

def pause_for_user(message="Press Enter to continue..."):
    """Pause the program until user presses Enter."""
    input(f"\n{message}")

def ask_user_input(prompt, default, cast_type=float):
    """
    Ask the user for input with a default value.
    If the user presses Enter without typing, the default is returned.

    Parameters:
        prompt(str): description of the variable
        default: default value
        cast_type: type to cast input (float or int)

    Returns:
        user_value: value entered by user or default
    """
    user_input = input(f"{prompt} [default = {default}]: ")
    if user_input.strip() == "":
        return default
    else:
        return cast_type(user_input)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

def baseline(x_train, y_train, x_test1, y_test1, x_test2, y_test2, threshold):
    """
    Baseline model using Linear Regression.
    
    Parameters:
        x_train(array): training feature array
        y_train(array): training output array
        x_test1(array): 1st test feature array
        y_test1(array): 1st test output array
        x_test2(array): 2nd test feature array
        y_test2(array): 2nd test output array
        threshold(float): Threshold for converting probabilities into binary predictions
    Returns:
        mse: Mean Squared Error
        acc: accuracy values
    """

    # Train linear regression
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)

    # Predict values
    y_train_pred = linreg.predict(x_train)
    y_test1_pred = linreg.predict(x_test1)
    y_test2_pred = linreg.predict(x_test2)

    # Convert to probabilities with sigmoid
    y_train_prob = sigmoid(y_train_pred)
    y_test1_prob = sigmoid(y_test1_pred)
    y_test2_prob = sigmoid(y_test2_pred)

    # Apply classification threshold
    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_test1_pred = (y_test1_prob >= threshold).astype(int)
    y_test2_pred = (y_test2_prob >= threshold).astype(int)

    # Calculate MSE
    mse_train = mean_squared_error(y_train, y_train_prob)
    mse_test1 = mean_squared_error(y_test1, y_test1_prob)
    mse_test2 = mean_squared_error(y_test2, y_test2_prob)

    # Get accuracy values
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test1 = accuracy_score(y_test1, y_test1_pred)
    acc_test2 = accuracy_score(y_test2, y_test2_pred)

    mse = {"train": mse_train, "test1": mse_test1, "test2": mse_test2}
    acc = {"train": acc_train, "test1": acc_test1, "test2": acc_test2}
    
    return mse, acc

def main():
    # --------------------------------
    # 1. Ask for parameters
    # --------------------------------

    print("\nSet training parameters (press Enter to use default values):\n")

    lrate = ask_user_input(
        "Learning rate (controls step size for gradient descent)", 
        default=0.01, cast_type=float
    )
    iterations = ask_user_input(
        "Number of training iterations (how many times the model updates weights)", 
        default=1000, cast_type=int
    )
    reg_strength = ask_user_input(
        "Regularization strength (helps prevent overfitting by shrinking weights)", 
        default=0.0, cast_type=float
    )
    threshold = ask_user_input(
        "Prediction threshold (cutoff for deciding between class 0 and 1)", 
        default=0.2, cast_type=float
    )

    print("\n Training will run with:")
    print(f"  Learning rate     = {lrate}")
    print(f"  Iterations        = {iterations}")
    print(f"  Regularization    = {reg_strength}")
    print(f"  Threshold         = {threshold}\n")

    pause_for_user("Parameters set. Press Enter to choose features...")

    # --------------------------------
    # 2a. load data
    # --------------------------------

    training_df = pd.read_csv("data/training.csv", sep = ";")
    test1_df = pd.read_csv("data/test-1.csv", sep=";")
    test2_df = pd.read_csv("data/test-2.csv", sep=";")

    # pause_for_user("Data loaded. Press Enter to preprocess...")

    # --------------------------------
    # 2b. Ask which features to use
    # --------------------------------
    all_features = [
        "age", "job", "marital", "education", "default", "balance", 
        "housing", "loan", "contact", "day", "month", "duration", 
        "campaign", "pdays", "previous", "poutcome"
    ]

    print("\nAvailable features:")
    for i, f in enumerate(all_features, start=1):
        print(f"{i}. {f}")

    feature_input = input("\nEnter the features to use as comma-separated list (leave blank to use all): ").strip()

    if feature_input == "":
        selected_features = all_features
    else:
        # Clean and validate user input
        selected_features = [f.strip() for f in feature_input.split(",")]
        invalid_features = [f for f in selected_features if f not in all_features]
        if invalid_features:
            print(f"Warning: The following features are invalid and will be ignored: {invalid_features}")
            selected_features = [f for f in selected_features if f in all_features]

    print(f"\nUsing features: {selected_features}\n")

    training_df = training_df[selected_features + ["y"]]
    test1_df = test1_df[selected_features + ["y"]]
    test2_df = test2_df[selected_features + ["y"]]

    pause_for_user("Features chosen. Press Enter to preprocess, train model, and show plots...")

    # --------------------------------
    # 3. preprocesses
    # --------------------------------

    # Identify categorical columns among selected features
    categorical_cols = [col for col in ["job", "marital", "education", "contact", "day", "month", "poutcome"] if col in selected_features]

    # Only one-hot encode if there are categorical columns selected
    if categorical_cols:
        training_df = pd.get_dummies(training_df, columns=categorical_cols)
        test1_df = pd.get_dummies(test1_df, columns=categorical_cols)
        test2_df = pd.get_dummies(test2_df, columns=categorical_cols)

        # Align test sets to training columns
        test1_df = test1_df.reindex(columns=training_df.columns, fill_value=0)
        test2_df = test2_df.reindex(columns=training_df.columns, fill_value=0)

    # convert yes/no -> 1/0 for binary columns (do separately)
    binary_cols = ["default", "housing", "loan", "y"]
    for col in binary_cols:
        if col in training_df.columns:
            training_df[col] = training_df[col].map({"yes": 1, "no": 0})
        if col in test1_df.columns:
            test1_df[col] = test1_df[col].map({"yes": 1, "no": 0})
        if col in test2_df.columns:
            test2_df[col] = test2_df[col].map({"yes": 1, "no": 0})

    # split features / labels
    x_train = training_df.drop("y", axis=1).values.astype(float)
    y_train = training_df["y"].values.reshape(-1, 1).astype(float)

    x_test1 = test1_df.drop("y", axis=1).values.astype(float)
    y_test1 = test1_df["y"].values.reshape(-1, 1).astype(float)

    x_test2 = test2_df.drop("y", axis=1).values.astype(float)
    y_test2 = test2_df["y"].values.reshape(-1, 1).astype(float)

    # Standardize features (fit on train, apply to tests)
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    x_train = (x_train - feature_mean) / feature_std
    x_test1 = (x_test1 - feature_mean) / feature_std
    x_test2 = (x_test2 - feature_mean) / feature_std

    # --------------------------------
    # 4. training
    # --------------------------------

    W_final, b_final, losses, history = training_logreg(
        x_train, y_train, lrate, iterations, reg_strength
        )

    # --------------------------------
    # 5. plot
    # --------------------------------

    # loss curve for model
    plt.figure(figsize=(8,5))
    plt.plot(range(len(losses)), losses, label="Training Loss", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve for Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/loss_curve.png", dpi=300)
    print("\nLoss curve saved as 'loss_curve.png'.")
    plt.show()

    # mean squared error for baseline (MSE seems to be different from loss in logistic regression so it can't be compared in the
    # same way)
    mse, acc = baseline(x_train, y_train, x_test1, y_test1, x_test2, y_test2, threshold)
    # plt.figure(figsize=(8,5))
    # plt.plot([mse["train"], mse["train"]], label="Training MSE", color="blue")
    # plt.xlabel("Iteration")
    # plt.ylabel("MSE")
    # plt.title("Baseline MSE")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    pause_for_user("Loss curve shown. Press Enter to plot accuracy...")

    # accuracy curve
    train_accuracies = []
    test1_accuracies = []
    test2_accuracies = []

    for (W_iter, b_iter) in history:
        # predictions at this iteration
        y_train_pred_iter = (sigmoid(x_train @ W_iter + b_iter) >= threshold).astype(int)
        y_test1_pred_iter = (sigmoid(x_test1 @ W_iter + b_iter) >= threshold).astype(int)
        y_test2_pred_iter = (sigmoid(x_test2 @ W_iter + b_iter) >= threshold).astype(int)

        train_accuracies.append(np.mean(y_train_pred_iter == y_train))
        test1_accuracies.append(np.mean(y_test1_pred_iter == y_test1))
        test2_accuracies.append(np.mean(y_test2_pred_iter == y_test2))

    # Creates 2 subplots with accuracy of both model and baseline side by side
    # Accuracy of the model
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].plot(range(len(losses)), train_accuracies, label="Training Accuracy", color="green")
    axes[0].plot(range(len(losses)), test1_accuracies, label="Test Set 1 Accuracy", color="orange")
    axes[0].plot(range(len(losses)), test2_accuracies, label="Test Set 2 Accuracy", color="red")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    
    # Accuracy of the baseline
    axes[1].plot([acc["train"], acc["train"]], label="Training Accuracy", color="green")
    axes[1].plot([acc["test1"], acc["test1"]], label="Test Set 1 Accuracy", color="orange")
    axes[1].plot([acc["test2"], acc["test2"]], label="Test Set 2 Accuracy", color="red")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Baseline Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("graphs/loss_curve.png", dpi=300)
    print("\nLoss curve saved as 'loss_curve.png'.")
    plt.show()

    pause_for_user("Accuracy curves shown. Press Enter to run final predictions and evaluations...")

    # --------------------------------
    # 6. predict
    # --------------------------------

    y_train_pred = predict(x_train, W_final, b_final, threshold)
    y_test1_pred = predict(x_test1, W_final, b_final, threshold)
    y_test2_pred = predict(x_test2, W_final, b_final, threshold)

    pause_for_user("Predictions complete. Press Enter to evaluate...")


    # --------------------------------
    
    # 7. evaluate
    # --------------------------------

    print("\nTraining finished. Now let's evaluate the model with different thresholds.")

    # Then: interactive loop for manual testing
    while True:
        threshold_input = input("\nEnter a threshold between 0 and 1 to evaluate (or press Enter to exit): ")
        if threshold_input.strip() == "":
            break  # exit loop
        try:
            threshold_value = float(threshold_input)
            print(f"\nEvaluating with threshold = {threshold_value:.2f}")

            # Predictions at chosen threshold
            y_train_pred = predict(x_train, W_final, b_final, threshold_value)
            y_test1_pred = predict(x_test1, W_final, b_final, threshold_value)
            y_test2_pred = predict(x_test2, W_final, b_final, threshold_value)

            print("\nTraining Performance:")
            evaluate_model(y_train, y_train_pred)

            print("\nTest Set 1 Performance:")
            evaluate_model(y_test1, y_test1_pred)

            print("\nTest Set 2 Performance:")
            evaluate_model(y_test2, y_test2_pred)

        except ValueError:
            print("Please enter a number between 0 and 1.")

    pause_for_user("\nAll done! Press Enter to exit.")


if __name__ == "__main__":
    main()