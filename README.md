This project trains and evaluates 24 neural network models using different hyperparameters. The goal is to see how different activation functions, learning rates, epochs, and hidden layer configurations affect model performance.

The dataset used: https://archive.ics.uci.edu/dataset/53/iris
Iris dataset has 150 samples of iris flowers across 3 different species. The dataset uses 4 features to classify these species --> sepal length, sepal width, petal length, and petal width.

We used MLPClassifier from scikit-learn to build and train all neural network models. We decided to have a fixed number of 10 neurons per hidden layer, with the number of hidden layers being varied as a hyperparameter.

To Run:
1) Download project files
2) Make sure Python libraries are installed: pip install numpy pandas matplotlib scikit-learn ucimlrepo
3) Run: python neural_net.py

Changes Made to the Code Outline:
1) Instead of loading the dataset from a local CSV file, we used the ucimlrepo library to fetch the dataset directly from the UCI ML Repository by ID, as we did in the last assignment.
2) The assignment seemed to suggest using MSE as the error metric, but the iris dataset is classification based, thus we chose log loss instead for a more accurate reading.

Code Output:
The program prints out a results table both to the terminal and to a csv file, containing each model's hyperparameters, training/test accuracy, and training/test log loss.
The program also prints out 3 loss curve plots, one for each activation function. 
