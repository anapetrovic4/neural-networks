import numpy as np
from tqdm import tqdm
import pandas as pd

"""
Implement a multilayer neural network with one hidden layer and an output layer
"""

class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        """
        Initialize weights randomly and set biases to zero.
        Implement Xavier initialization to keep balance between input and output variance.
        """
        self.input_hidden_weights = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size) # objasni
        self.output_hidden_weights = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size) # objasni
        
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.activation = activation

    # Implement 2-3 activation functions (relu, sigmoid, tanh, softmax...)
    """
    Tanh activation function
    """
    def tanh(self, x):
        return np.tanh(x)
    
    def dtanh(self, x): # Derivative
        return 1 - np.tanh(x) ** 2
    """
    Softmax activation function transforms outputs of the neural network
    into a vector of probabilities - a probability distribution ot the input classes. 
    """
    def softmax(self, x):
        exp_x = np.exp(x-np.max(x, axis=1, keepdims=True)) 
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    """
    Rectified Linear Unit (ReLU) activation function returns 0 if the input is negative,
    but for any non-negative value it returns than value. 
    """
    def ReLU(self, x): 
        return np.maximum(0, x)
    
    def dReLU(self, x): # Derivative 
        return (x > 0).astype(float)

    """
    Implement helpers functions for (de)activating 
    activation functions in forward and backward propagation.
    """
    def activate(self, x):
        if self.activation == 'tanh':
            return self.tanh(x)
        elif self.activation == 'relu':
            return self.ReLU(x)
    def deactivate(self, x):
        if self.activation == 'tanh':
            return self.dtanh(x)
        elif self.activation == 'relu':
            return self.dReLU(x)

    """
    Computation of hidden input and output layers 
    with application of activation functions.
    """
    def forward(self, X): 
        self.hidden_input = np.dot(X, self.input_hidden_weights) + self.bias_hidden
        self.hidden_output = self.activate(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.output_hidden_weights) + self.bias_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    """
    Computation of errors and adaptation of weights and biases
    using the gradient descent algorithm.
    """
    def backward(self, X, y, output, lr):
        output_error = output - y 

        hidden_error = np.dot(output_error, self.output_hidden_weights.T) * self.deactivate(self.hidden_input)

        self.output_hidden_weights -= lr * np.dot(self.hidden_output.T, output_error) 
        self.bias_output -= lr * np.sum(output_error, axis=0, keepdims=True)
        self.input_hidden_weights -= lr * np.dot(X.T, hidden_error)
        self.bias_hidden -= lr * np.sum(hidden_error, axis=0, keepdims=True) 

    """
    Perform forward and backward propagation for a specified number of epochs and print the loss periodically.
    """
    def train(self, X, y, epochs, lr):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        for epoch in tqdm(range(epochs), desc="Training"):
            output = self.forward(X)
            self.backward(X, y, output, lr)

            if (epoch+1) % 100 == 0 or epoch == 0:
                loss = -np.sum(y * np.log(output + 1e-8)) / X.shape[0] #Cross-entropy loss
                loss = loss.item() # convert Series to float
                print(f"Epoch {epoch+1}, Loss {loss:.4f}")

    """
    Make predictions
    """
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    """
    Evaluate the neural network
    """
    def evaluate_accuracy(self, X, y_true):
        y_pred = self.predict(X)
        y_true_labels = np.argmax(y_true, axis=1)
        acc = np.mean(y_pred == y_true_labels)
        return acc

    """
    Implement a function that calculates number of parameters and RAM usage
    """
    def parameters_and_memory(self):
        params = (
            np.prod(self.input_hidden_weights.shape) +
            np.prod(self.output_hidden_weights.shape) +
            np.prod(self.bias_hidden.shape) +
            np.prod(self.bias_output.shape)
        )

        dtype = self.input_hidden_weights.dtype
        bytes_parameter = np.dtype(dtype).itemsize # 8 bytes per parameter (float64)
        total_memory_bytes = params * bytes_parameter

        print(f"Total learnable parameters: {params}")
        print(f"Each parameter dtype: {dtype} ({bytes_parameter} bytes)")
        print(f"Estimated memory usage: {total_memory_bytes / 1024:.2f} KB ({total_memory_bytes / (1024**2):.2f} MB)")

"""
Load the datasets 
"""
X_train = pd.read_csv('../datasets/iris_train.csv')
y_train = pd.read_csv('../datasets/iris_train_labels.csv')
X_test = pd.read_csv('../datasets/iris_test.csv')
y_test = pd.read_csv('../datasets/iris_test_labels.csv')

"""
Implement simple method to search for the best hyperparameters (gridsearch)
"""
def grid_search(X_train, y_train, X_test, y_test):
    hidden_sizes = [4, 8, 16]
    activations = ['tanh', 'relu']
    epochs = [100, 500, 1000]
    lrs = [0.01, 0.05, 0.1]

    best_acc = 0
    best_config = None

    for h in hidden_sizes:
        for act in activations:
            for epoch in epochs:
                for lr in lrs:
                    print(f"Training with hidden_size={h}, activation={act}, lr={lr}, epochs={epoch}")
                    mlp = MLP(input_size=4, hidden_size=h, output_size=3, activation=act)
                    mlp.parameters_and_memory()
                    mlp.train(X_train, y_train, epochs=epoch, lr=lr)
                    acc = mlp.evaluate_accuracy(X_test, y_test)
                    print(f"Accuracy: {acc:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        best_config = (h, act, lr, epoch)
    print(f"\nBest accuracy {best_acc:.4f} with config: hidden_size={best_config[0]}, activation={best_config[1]}, lr={best_config[2]}, epochs={best_config[3]}")

grid_search(X_train, y_train, X_test, y_test)