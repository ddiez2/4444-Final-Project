import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from mlp import Relu, Linear, SquaredError, MultilayerPerceptron, Layer, Mish
import matplotlib.pyplot as plt
from tabulate import tabulate
import random

def load_auto_mpg(data_path: str = "../MPG/auto-mpg.data"):
    """
    MOST OF THE CODE IS FROM THE SAMPLE CODE GIVEN FROM EXAMPLE NOTEBOOK LINK.
    MY CODE IS COMMENTED WITH AN ALIKE HEADING.
    """
    column_names = [
        "mpg", "cylinders", "displacement", "horsepower", "weight", 
        "acceleration", "model_year", "origin", "car_name"
    ]

    # Load dataset, ensuring we ignore the car name column
    df = pd.read_csv(data_path, sep='\s+', names=column_names, na_values="?", usecols=range(8))

    # Drop rows with missing values
    df = df.dropna()

    # Convert dataframe to numpy array (only numeric columns)
    data = df.values

    # Separate input features and target variable
    X = data[:, 1:].astype(float)  # Features (all columns except mpg)
    y = data[:, 0].astype(float).reshape(-1, 1)  # Target (mpg), reshaped to (n, 1)

    # Normalize features
    scaler = StandardScaler()
    X = X / np.mean(X, axis=0)
    y = y / np.mean(y, axis=0)
    # Split dataset (70% train, 15% validation, 15% test)
    train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.3, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

    # Return without transposing - shapes will be (n_samples, n_features) for X and (n_samples, 1) for y
    return train_x, train_y, val_x, val_y, test_x, test_y


def build_mpg_model():
    """
    CREATES A CUSTOM MODEL WITH MISH AND RELU ACTIVATION FUNCTION.
    LAYERS ARE IN THE ORDER OF 7-32-16-8-1.
    LAST LAYER IS A LINEAR LAYER FOR REGRESSION.
    """
    layers = [
        Layer(fan_in=7, fan_out=32, activation_function=Mish(), dropout_rate=0.1),
        Layer(fan_in=32, fan_out=16, activation_function=Relu(), dropout_rate=0.1),
        Layer(fan_in=16, fan_out=8, activation_function=Relu(), dropout_rate=0.1),
        Layer(fan_in=8, fan_out=1, activation_function=Linear())  # Linear output for regression
    ]
    return MultilayerPerceptron(layers)

def plot_loss_curves(training_losses, validation_losses):
    """
    PLOTS THE LOSS CURVES FOR TRAINING AND VALIDATION.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves for Auto MPG Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig('MPG_loss_curves.png')
    plt.show()  
    plt.close()

def report_test_predictions(model, test_x, test_y, num_samples=10):
    """
    Generate a report of predicted vs actual MPG for random test samples.
    """
    indices = random.sample(range(len(test_x)), num_samples)
    
    # Get predictions for selected samples
    predictions = []
    actuals = []
    for idx in indices:
        # Reshape and transpose the input data correctly
        input_data = test_x[idx:idx+1].T  # Transpose to shape (7,1)
        pred = model.forward(input_data)[0][0]
        actual = test_y[idx][0]
        predictions.append(pred)
        actuals.append(actual)
    
    # Create table data
    table_data = []
    for i in range(num_samples):
        table_data.append([
            i+1,
            f"{actuals[i]:.2f}",
            f"{predictions[i]:.2f}",
            f"{abs(predictions[i] - actuals[i]):.2f}"
        ])
    
    # Generate table
    headers = ["Sample", "Actual MPG", "Predicted MPG", "Absolute Error"]
    return tabulate(table_data, headers=headers, tablefmt="grid")

def train_and_evaluate_mpg(epochs, batch_size, learning_rate, use_rmsprop):
    """
    CODE TO TRAIN AND EVALUATE THE MODEL ON AUTO MPG DATASET.
    PRINTS THE TEST RESULTS AND PREDICTION REPORT.
    """
    train_x, train_y, val_x, val_y, test_x, test_y = load_auto_mpg()

    mlp = build_mpg_model()
    loss_func = SquaredError()

    training_losses, validation_losses = mlp.train(
        train_x, train_y, val_x, val_y, loss_func,
        learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, rmsprop=use_rmsprop
    )

    plot_loss_curves(training_losses, validation_losses)
    test_loss = mlp.test(test_x, test_y, loss_func)
    print("\nTest Results:")
    print(f"Total Testing Loss: {test_loss:.4f}")
    print("\nPrediction Samples:")
    print(report_test_predictions(mlp, test_x, test_y))

    return mlp, training_losses, validation_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on Auto MPG dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--rmsprop", action="store_true", help="Use RMSProp optimizer")
    args = parser.parse_args()

    mlp, train_losses, val_losses = train_and_evaluate_mpg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_rmsprop=args.rmsprop
    )
