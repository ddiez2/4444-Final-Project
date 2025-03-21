import numpy as np
import struct
from array import array
from os.path import join, exists
import argparse
from sklearn.model_selection import train_test_split
from mlp import Relu, Softmax, CrossEntropy, MultilayerPerceptron, Layer, Mish
import matplotlib.pyplot as plt

'''
MOST CODE FOR INGESTION IS FROM THE SAMPLE CODE GIVEN FROM MOODLE LINK.
MY CODE IS COMMENTED WITH AN ALIKE HEADING.
'''
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch in labels file, expected 2049, got {magic}")
            labels = array("B", file.read())

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch in images file, expected 2051, got {magic}")
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            # Get one image and reshape it to 28x28
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype=np.float32)
            img = img.reshape(rows, cols)
            images.append(img)
        return images, labels

    def load_data(self):
        train_x, train_y = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, test_y = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (train_x, train_y), (x_test, test_y)

def one_hot_encode(labels, num_classes=10):
    labels = np.array(labels)
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def build_mnist_model():
    """
    CREATES A CUSTOM MODEL WITH MISH ACTIVATION FUNCTION.
    LAYERS ARE IN THE ORDER OF 784-128-64-10.
    LAST LAYER IS A SOFTMAX LAYER FOR CLASSIFICATION.
    """
    layers = [
        Layer(fan_in=784, fan_out=128, activation_function=Mish(), dropout_rate=0.1),
        Layer(fan_in=128, fan_out=64, activation_function=Mish(), dropout_rate=0.1),
        Layer(fan_in=64, fan_out=10, activation_function=Softmax())  # Output layer with Softmax for classification
    ]
    return MultilayerPerceptron(layers)

def plot_loss_curves(training_losses, validation_losses):
    """PLOT LOSS CURVES FOR TRAINING AND VALIDATION DATA."""
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves for MNIST Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig('mnist_loss_curves.png')
    plt.show()
    plt.close()

def show_sample_predictions(x_test, test_y, pred_y):
    """
    DISPLAYS SAMPLE PREDICTIONS FOR THE MNIST DATASET.
    """
    predicted_labels = np.argmax(pred_y, axis=0)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for digit in range(10):
        idx = np.where(test_y == digit)[0][0]
        img = x_test[idx].reshape(28, 28)
        
        axes[digit].imshow(img, cmap='gray')
        axes[digit].axis('off')
        axes[digit].set_title(f'True: {digit}\nPred: {predicted_labels[idx]}')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    plt.show()
    plt.close()

def main(epochs, batch_size, learning_rate, use_rmsprop):
    '''
    Follows sample code given on Moodle.
    '''
    input_path = "../MNIST/"
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    for f in [training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath]:
        if not exists(f):
            raise FileNotFoundError(f"File not found: {f}. Please download the MNIST dataset into the ../MNIST directory.")

    dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                 test_images_filepath, test_labels_filepath)
    (train_x, train_y), (x_test, test_y) = dataloader.load_data()

    train_x = np.array(train_x)  # shape: (n_samples, 28, 28)
    x_test = np.array(x_test)    # shape: (n_samples, 28, 28)
    train_x = train_x / 255.0 # Normalize pixel values to [0, 1]
    x_test = x_test / 255.0  # Normalize pixel values to [0, 1]

    train_x = train_x.reshape(train_x.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=25)
    train_y = one_hot_encode(train_y, num_classes=10)
    val_y = one_hot_encode(val_y, num_classes=10)
    '''
    MY CODE STARTS HERE.
    TRAIN AND TEST THE MODEL ON THE MNIST DATASET.
    '''
    mlp = build_mnist_model()
    loss_func = CrossEntropy()

    training_losses, validation_losses = mlp.train(
        train_x, train_y, val_x, val_y, loss_func,
        learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, rmsprop=use_rmsprop
    )

    plot_loss_curves(training_losses, validation_losses) # Plot loss curves

    pred_y = mlp.forward(x_test.T, training=False)  # Output shape: (10, n_samples)
    predicted_labels = np.argmax(pred_y, axis=0)       # Predicted digit for each test sample

    test_y = np.array(test_y) 
    accuracy = np.mean(predicted_labels == test_y) # Calculate accuracy
    print("\nTest Results:")
    print(f"Total Testing Accuracy: {accuracy * 100:.2f}%")

    show_sample_predictions(x_test, test_y, pred_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on MNIST dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--rmsprop", action="store_true", help="Use RMSProp optimizer")
    args = parser.parse_args()

    main(epochs=args.epochs, batch_size=args.batch_size,
         learning_rate=args.learning_rate, use_rmsprop=args.rmsprop)