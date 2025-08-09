#Update, this doesnt seem completely right so i have a more stable but older model that shows off how it learns better

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, colormap="viridis"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.colormap = colormap  # Use your custom color system via a colormap

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Input to hidden layer
        self.b1 = np.zeros((1, hidden_size))  # Hidden layer bias
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Hidden to output layer
        self.b2 = np.zeros((1, output_size))  # Output layer bias

        # Initialize learning space (size matching output)
        self.learning_space = np.zeros((hidden_size, output_size))

    def forward(self, X):
        # Ensure X is 2D
        X = X.reshape(1, -1)  # Reshape input X to have shape (1, input_size)

        # Forward pass
        self.a1 = np.dot(X, self.W1) + self.b1  # Hidden layer linear combination
        self.z1 = np.tanh(self.a1)  # Using tanh as activation function
        self.a2 = np.dot(self.z1, self.W2) + self.b2  # Output layer linear combination
        self.output = self.a2  # No activation at the output (regression style)
        return self.output

    def backward(self, X, y, output):
        # Ensure X is 2D
        X = X.reshape(1, -1)  # Reshape input X to have shape (1, input_size)

        # Compute gradients using Mean Squared Error loss derivative
        dloss_doutput = output - y
        dloss_dW2 = np.dot(self.z1.T, dloss_doutput)  # Gradient wrt W2
        dloss_db2 = np.sum(dloss_doutput, axis=0, keepdims=True)  # Gradient wrt b2

        dloss_dz1 = np.dot(dloss_doutput, self.W2.T) * (1 - np.tanh(self.a1) ** 2)  # Backprop through tanh
        dloss_dW1 = np.dot(X.T, dloss_dz1)  # Gradient wrt W1
        dloss_db1 = np.sum(dloss_dz1, axis=0, keepdims=True)  # Gradient wrt b1

        # Gradient clipping to avoid exploding gradients
        max_grad_value = 1.0
        dloss_dW1 = np.clip(dloss_dW1, -max_grad_value, max_grad_value)
        dloss_db1 = np.clip(dloss_db1, -max_grad_value, max_grad_value)
        dloss_dW2 = np.clip(dloss_dW2, -max_grad_value, max_grad_value)
        dloss_db2 = np.clip(dloss_db2, -max_grad_value, max_grad_value)

        # Update weights and biases using gradient descent
        learning_rate = 0.001  # Reduced learning rate for stability
        self.W1 -= learning_rate * dloss_dW1
        self.b1 -= learning_rate * dloss_db1
        self.W2 -= learning_rate * dloss_dW2
        self.b2 -= learning_rate * dloss_db2

        # Update learning space dynamically based on perceptron activity and error
        self.update_learning_space(X, dloss_doutput)

    def update_learning_space(self, X, dloss_doutput):
        """
        Update the learning space based on perceptron activity and error.
        Higher error results in more significant updates to the learning space.
        """
        perceptron_activity = self.z1  # Activity from the hidden layer
        error_significance = np.abs(dloss_doutput)  # Measure of error magnitude at output

        # Adjust learning space based on perceptron activity and error significance
        adjustment = np.dot(perceptron_activity.T, error_significance) * 0.01  
        self.learning_space += adjustment

    def visualize_learning_space(self):
        """
        Visualize the learning space using the provided color system.
        """
        plt.imshow(self.learning_space, cmap=self.colormap, interpolation='nearest')
        plt.title("Learning Space Visualization")
        plt.colorbar()
        plt.show()

    def train(self, samples, epochs=1000, vis_interval=100000):
        """
        Train the network using a list of samples.
        Each sample is a tuple: (np.array([a, b]), task, one_hot_label)
        """
        for epoch in range(epochs):
            # Randomly select a sample from the dataset
            X_sample, task, label = random.choice(samples)

            # Compute the expected result and select operator symbol for display
            if task == "addition":
                result = X_sample[0] + X_sample[1]
                op = "+"
            elif task == "subtraction":
                result = X_sample[0] - X_sample[1]
                op = "-"
            elif task == "multiplication":
                result = X_sample[0] * X_sample[1]
                op = "*"
            elif task == "division":
                result = X_sample[0] // X_sample[1]
                op = "/"
            else:
                result = None
                op = "?"

            print(f"Epoch {epoch}/{epochs}, Task: {task}, Problem: {X_sample[0]} {op} {X_sample[1]} = {result}")

            output = self.forward(X_sample)
            self.backward(X_sample, label, output)

            # Visualize learning space at specified intervals
            if epoch % vis_interval == 0:
                loss = np.mean((output - label) ** 2)
                print(f"Loss: {loss}")
                self.visualize_learning_space()

# One-Hot Encoding utility function
def create_one_hot_encoding(result, max_value=100):
    """
    Create a one-hot encoded vector for a given result.
    The result is assumed to be an integer between 1 and max_value (inclusive).
    """
    # Note: scikit-learn’s OneHotEncoder has changed its API;
    # here we simply build a one-hot vector manually.
    one_hot = np.zeros(max_value)
    # Adjust index: result of 1 corresponds to index 0, etc.
    if 1 <= result <= max_value:
        one_hot[result - 1] = 1
    else:
        raise ValueError(f"Result {result} out of bounds for one-hot encoding with size {max_value}")
    return one_hot

# Generate a math dataset containing addition, subtraction, multiplication, and division samples.
def generate_math_dataset(num_samples_per_task=1000, min_val=1, max_val=10):
    """
    Returns a list of samples.
    Each sample is a tuple: (np.array([a, b]), task, one_hot_encoded_answer)
    For subtraction, we ensure a >= b (so result >= 0) and then add 1 to the result to index properly.
    For division, we generate pairs where a is an exact multiple of b.
    """
    samples = []

    # Addition samples: result = a + b (range: 2 to 20)
    for _ in range(num_samples_per_task):
        a = np.random.randint(min_val, max_val+1)
        b = np.random.randint(min_val, max_val+1)
        result = a + b
        label = create_one_hot_encoding(result, max_value=max_val**2)
        samples.append((np.array([a, b]), "addition", label))

    # Subtraction samples: ensure a >= b so that result >= 0.
    # We then add 1 to the result so that a 0 result is encoded at index 0.
    for _ in range(num_samples_per_task):
        a = np.random.randint(min_val, max_val+1)
        b = np.random.randint(min_val, a+1)  # b <= a
        result = a - b
        label = create_one_hot_encoding(result + 1, max_value=max_val**2)
        samples.append((np.array([a, b]), "subtraction", label))

    # Multiplication samples: result = a * b (range: 1 to 100)
    for _ in range(num_samples_per_task):
        a = np.random.randint(min_val, max_val+1)
        b = np.random.randint(min_val, max_val+1)
        result = a * b
        label = create_one_hot_encoding(result, max_value=max_val**2)
        samples.append((np.array([a, b]), "multiplication", label))

    # Division samples: generate pairs so that a is divisible by b.
    # We choose b randomly then pick a multiplier.
    for _ in range(num_samples_per_task):
        b = np.random.randint(min_val, max_val+1)
        multiplier = np.random.randint(1, max_val+1)
        a = b * multiplier  # ensures a is exactly divisible by b
        result = a // b  # integer division
        label = create_one_hot_encoding(result, max_value=max_val**2)
        samples.append((np.array([a, b]), "division", label))

    return samples

# --- Initialize and train the network ---

# Set parameters: 
input_size = 2           # two numbers as input
hidden_size = 64         # you may experiment with this size
output_size = 100        # Using 100 as the maximum output range (for 10*10, etc.)
nn = NeuralNetwork(input_size, hidden_size, output_size, colormap="viridis")  # or use your favorite colormap

# Generate the dataset (each task gets num_samples_per_task examples)
samples = generate_math_dataset(num_samples_per_task=1000, min_val=1, max_val=10)

# Train the network.
# Here we use a training loop that randomly selects one sample at a time.
nn.train(samples, epochs=5000000, vis_interval=100000)
