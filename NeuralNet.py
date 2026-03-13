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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
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

        print(len(self.processed_data))
        print(self.processed_data.head)

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("train.csv") # put in path to your file
    neural_network.preprocess()
   # neural_network.train_evaluate()
