#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from ucimlrepo import fetch_ucirepo

class NeuralNet:
    def __init__(self, dataFile, header=True):
        # Fetch Iris dataset from UCI
        iris = fetch_ucirepo(id=53)

        # Features and targets
        X = iris.data.features
        y = iris.data.targets

        # Combine into one dataframe like your original code expects
        self.raw_input = pd.concat([X, y], axis=1)

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    # categorical to numerical, etc
    def preprocess(self):
        #self.processed_data = self.raw_input
        df = self.raw_input.copy()

        # Handle null values (drop rows with nulls)
        df.dropna(inplace=True)

        # Encode species labels
        label_map = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        df['class'] = df['class'].map(label_map).astype(int)

        # Split features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Store processed data
        self.processed_data = pd.DataFrame(X_scaled)
        self.processed_data[df.columns[-1]] = y.values

        #print(len(self.processed_data))
        #print(self.processed_data.head)

        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]

        # split into train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Fixed neurons per hidden layer
        neurons_per_layer = 10

        results = []  # will hold one dict per model

        # We'll collect per-epoch loss curves for each model
        history = {}   # key -> list of loss values

        combos = list(product(activations, learning_rate, max_iterations, num_hidden_layers))
        total  = len(combos)
        print(f"Training {total} models: 8 with logistic, 8 with tanh, and 8 with relu...\n")

        for idx, (activation_func, learn_rate, epochs, n_hidden_layers) in enumerate(combos, 1):
            hidden_layer_sizes = tuple([neurons_per_layer] * n_hidden_layers)

            # create a label for easy reading
            label = f"act={activation_func}, lr={learn_rate}, ep={epochs}, layers={n_hidden_layers}"
            print(f"[{idx}/{total}] {label}")

            # Create the neural network and be sure to keep track of the performance metrics
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation_func,
                learning_rate_init=learn_rate,
                max_iter=epochs,
                random_state=42,
                n_iter_no_change=epochs   # prevent early stopping
            )

            # Train Model
            model.fit(X_train, y_train)

            # store the epoch loss to create the loss curve later on
            history[label] = model.loss_curve_
 
            # get the overall prediction/output for both train and test for this model
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)
 
            # get the accuracy scores and errors (log loss)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc  = accuracy_score(y_test,  y_test_pred)
            train_loss = log_loss(y_train, model.predict_proba(X_train))
            test_loss  = log_loss(y_test,  model.predict_proba(X_test))
 
            # append to results
            results.append({
                "Activation": activation_func,
                "Learning Rate": learn_rate,
                "Max Epochs": epochs,
                "Hidden Layers": n_hidden_layers,
                "Train Accuracy": round(train_acc, 4),
                "Test Accuracy": round(test_acc,  4),
                "Train Log Loss": round(train_loss, 4),
                "Test Log Loss": round(test_loss,  4),
            })

        # Convert Results 
        results_df = pd.DataFrame(results)

        print("\n Model Results")
        print(results_df)

        # Save table
        results_df.to_csv("model_results.csv", index=False)

        # Plot the model history for each model in a single plot
        for label, loss_values in history.items():
            plt.figure(figsize=(8,6))
            plt.plot(loss_values, marker='o')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training Loss vs Epochs\n{label}")
            plt.grid(True)
            
            # Create file label
            file_label = label.replace("=", "").replace(",", "").replace(" ", "_")
            filename = f"{file_label}.png"
            
            plt.savefig(filename)
            plt.close()
    
        # model history is a plot of accuracy (MSE) vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.
       
        return 0

if __name__ == "__main__":
    neural_network = NeuralNet("train.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()

