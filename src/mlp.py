import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n_samples, n_features)
    :param train_y (np.ndarray): Target values of shape (n_samples, 1)
    :param batch_size (int): The size of each batch.
    :return: tuple of (batch_x.T, batch_y.T) with shapes (n_features, batch_size) and (1, batch_size)
    """
    samples = train_x.shape[0] # Number of samples
    indicies = np.arange(samples) # Create an array of indices
    np.random.shuffle(indicies) # Shuffle the indices

    for i in range(0, samples, batch_size):
        batch_indicies = indicies[i:i+batch_size]
        batch_x = train_x[batch_indicies]
        batch_y = train_y[batch_indicies]
        # Transpose the batches to match the expected shapes for the network
        yield batch_x.T, batch_y.T

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x) * (1 - self.forward(x))

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        softplus = np.log(1 + np.exp(x))
        tanh_sp = np.tanh(softplus)
        return np.exp(x) * (tanh_sp + x * (1 - tanh_sp ** 2)) / (1 + np.exp(x))

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-8
        return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=0))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true




class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float=0.0):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        :param dropout_rate: dropout rate for this layer
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the z term (pre-activation values) from diagram
        self.z = None
        #this will store the dropout mask (randomly drop neurons)
        self.dropout_mask = None

        # Initialize weights and biaes
        #Glorot initialization
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_out, fan_in))  # weights
        self.b = np.zeros((fan_out, 1))  # biases

    def forward(self, h: np.ndarray, training: bool=True) -> np.ndarray:
        """
        Computes the activations for this layer

        :param h: input to layer
        :param training: boolean flag to indicate if we are training
        :return: layer activations
        """
        self.z = np.dot(self.W, h) + self.b # pre-activation values (z = Wx + b)
        self.activations = self.activation_function.forward(self.z) # activations (h = f(z)) arbitrary activation function
        
        # Apply dropout (only during training)
        if training and self.dropout_rate > 0:
            # Create binary dropout mask with proper scaling
            self.dropout_mask = (np.random.random(self.activations.shape) > self.dropout_rate)
            # Scale the mask to maintain expected values
            self.dropout_mask = self.dropout_mask.astype(float) / (1.0 - self.dropout_rate)
            # Apply mask using multiplication
            self.activations = np.multiply(self.activations, self.dropout_mask)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # For Softmax layer, bypass multiplication by derivative (assuming loss derivative is y_pred - y_true)
        if isinstance(self.activation_function, Softmax):
            self.delta = delta
        else:
            activation_derivative = self.activation_function.derivative(self.z)
            self.delta = delta * activation_derivative

        # If dropout was applied during forward, apply the same mask to the gradients
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            self.delta = self.delta * self.dropout_mask

        dL_dW = np.dot(self.delta, h.T) # gradient of loss w.r.t. weights
        dL_db = np.sum(self.delta, axis=1, keepdims=True) # gradient of loss w.r.t. biases
        delta_prev = np.dot(self.W.T, self.delta) # delta to pass to previous layer
        return dL_dW, dL_db, delta_prev




class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input of shape (n_features, batch_size)
        :param training: boolean flag to indicate if we are training (applies dropout)
        :return: network output of shape (n_outputs, batch_size)
        """
        h = x  # input to first layer is the input data
        for layer in self.layers:
            h = layer.forward(h, training) # compute activations for each layer
        return h

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (Tuple of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = [] 
        dl_db_all = []

        delta = loss_grad # start with the gradient of the loss function

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            h = input_data if i == 0 else self.layers[i - 1].activations # input data for the first layer, otherwise the activations of the previous layer
            dl_dw, dl_db, delta = layer.backward(h, delta) # compute gradients for each layer
            dl_dw_all.insert(0, dl_dw)
            dl_db_all.insert(0, dl_db) 

        return dl_dw_all, dl_db_all   # return the gradients

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, 
              loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, 
              rmsprop: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :param rmsprop: boolean flag to indicate if we should use RMSProp
        :return: (training_losses, validation_losses)
        """
        training_losses = []
        validation_losses = []
        beta = 0.9  
        epsilon = 1e-8  
        v_W = [np.zeros_like(layer.W) for layer in self.layers]   # RMSProp memory for weights
        v_b = [np.zeros_like(layer.b) for layer in self.layers]   # RMSProp memory for biases
        for epoch in range(epochs):
            batch_losses = []

            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size): # iterate over batches
                y_pred = self.forward(batch_x, training=True) # forward pass
                loss_grad = loss_func.derivative(batch_y, y_pred) # compute loss gradient
                dL_dW_all, dL_db_all = self.backward(loss_grad, batch_x) # backward pass

                for i, layer in enumerate(self.layers): # update weights and biases
                    if rmsprop: # RMSProp update
                        v_W[i] = beta * v_W[i] + (1 - beta) * dL_dW_all[i] ** 2
                        v_b[i] = beta * v_b[i] + (1 - beta) * dL_db_all[i] ** 2
                        layer.W -= (learning_rate / np.sqrt(v_W[i] + epsilon)) * dL_dW_all[i]
                        layer.b -= (learning_rate / np.sqrt(v_b[i] + epsilon)) * dL_db_all[i]
                    else: # Vanilla update
                        layer.W -= learning_rate * dL_dW_all[i]
                        layer.b -= learning_rate * dL_db_all[i]

                batch_losses.append(loss_func.loss(batch_y, y_pred)) # compute loss for this batch

            training_losses.append(np.mean(batch_losses)) # compute average loss for this epoch
            
            # Transpose validation data for forward pass
            val_pred = self.forward(val_x.T, training=False) 
            val_loss = loss_func.loss(val_y.T, val_pred)
            validation_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {training_losses[-1]:.4f}, Validation Loss: {validation_losses[-1]:.4f}")

        return np.array(training_losses), np.array(validation_losses) 
    
    def test(self, test_x: np.ndarray, test_y: np.ndarray, loss_func: LossFunction) -> np.ndarray:
        """
        Test the multilayer perceptron

        :param test_x: full test set input of shape (n x d) n = number of samples, d = number of features
        :param test_y: full test set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param loss_func: instance of a LossFunction
        :return: loss on test set
        """
        y_pred = self.forward(test_x.T, training=False) # forward pass
        return loss_func.loss(test_y.T, y_pred) # compute loss