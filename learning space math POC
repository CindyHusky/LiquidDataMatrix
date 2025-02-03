import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm  # progress bar

# -----------------------------
# Neural Network Definition
# -----------------------------
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, colormap="viridis"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.colormap = colormap

        # Initialize weights and biases (small random values)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Restore original (doubled) learning space size: shape = (hidden_size*2, output_size*2)
        self.learning_space = np.zeros((hidden_size * 2, output_size * 2))

    def forward(self, X):
        """
        Forward pass. Supports batch input of shape (batch_size, input_size)
        """
        self.a1 = np.dot(X, self.W1) + self.b1         # (batch_size, hidden_size)
        self.z1 = np.tanh(self.a1)                       # hidden layer activation
        self.a2 = np.dot(self.z1, self.W2) + self.b2     # (batch_size, output_size)
        return self.a2  # raw outputs

    def backward_batch(self, X, y, output):
        """
        Backward pass for a batch of samples.
        """
        batch_size = X.shape[0]
        dloss_doutput = (output - y)  # derivative of MSE loss

        # Gradients for output layer weights and biases
        dloss_dW2 = np.dot(self.z1.T, dloss_doutput) / batch_size
        dloss_db2 = np.sum(dloss_doutput, axis=0, keepdims=True) / batch_size

        # Backprop through tanh activation: derivative = 1 - tanh^2(a1)
        dloss_dz1 = np.dot(dloss_doutput, self.W2.T) * (1 - np.tanh(self.a1) ** 2)
        dloss_dW1 = np.dot(X.T, dloss_dz1) / batch_size
        dloss_db1 = np.sum(dloss_dz1, axis=0, keepdims=True) / batch_size

        # Clip gradients to avoid exploding gradients
        max_grad_value = 1.0
        dloss_dW1 = np.clip(dloss_dW1, -max_grad_value, max_grad_value)
        dloss_db1 = np.clip(dloss_db1, -max_grad_value, max_grad_value)
        dloss_dW2 = np.clip(dloss_dW2, -max_grad_value, max_grad_value)
        dloss_db2 = np.clip(dloss_db2, -max_grad_value, max_grad_value)

        # Update weights and biases using gradient descent
        learning_rate = 0.001
        self.W1 -= learning_rate * dloss_dW1
        self.b1 -= learning_rate * dloss_db1
        self.W2 -= learning_rate * dloss_dW2
        self.b2 -= learning_rate * dloss_db2

        # Update the learning space based on this batch
        self.update_learning_space_batch(X, dloss_doutput)

    def update_learning_space_batch(self, X, dloss_doutput):
        """
        Update the learning space based on the average hidden activity and error from a batch.
        """
        # Average hidden activity over the batch (shape: (1, hidden_size))
        perceptron_activity_avg = np.mean(self.z1, axis=0, keepdims=True)
        # Average absolute error over the batch (shape: (1, output_size))
        error_significance_avg = np.mean(np.abs(dloss_doutput), axis=0, keepdims=True)
        # Compute adjustment (shape: (hidden_size, output_size))
        adjustment = np.dot(perceptron_activity_avg.T, error_significance_avg) * 0.01
        # Tile the adjustment to double its dimensions (Kronecker product with a 2x2 ones matrix)
        doubled_adjustment = np.kron(adjustment, np.ones((2, 2)))
        # Update the learning space
        self.learning_space += doubled_adjustment

    def visualize_learning_space(self):
        plt.imshow(self.learning_space, cmap=self.colormap, interpolation='nearest')
        plt.title("Learning Space Visualization")
        plt.colorbar()
        plt.show()

    def train(self, samples, epochs=500000, batch_size=32, vis_interval=1000):
        num_samples = len(samples)
        # Use tqdm for a progress bar
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            # Select a random batch of samples
            batch_indices = np.random.choice(num_samples, batch_size, replace=False)
            X_batch_list = []
            y_batch_list = []
            details_batch = []  # For logging an example problem

            for idx in batch_indices:
                X_sample, task, label = samples[idx]
                # Ensure arithmetic problems have 3 numbers (pad if necessary)
                if X_sample.shape[0] == 2:
                    X_sample = np.append(X_sample, 0)
                X_batch_list.append(X_sample)
                y_batch_list.append(label)

                # Build a description string for logging purposes
                if task == "algebra":
                    a, b, c = X_sample
                    result = (c - b) // a  # exact integer solution
                    details_batch.append(f"{a}x + {b} = {c}, x = {result}")
                elif task == "addition":
                    result = X_sample[0] + X_sample[1]
                    details_batch.append(f"{X_sample[0]} + {X_sample[1]} = {result}")
                elif task == "subtraction":
                    result = X_sample[0] - X_sample[1]
                    details_batch.append(f"{X_sample[0]} - {X_sample[1]} = {result}")
                elif task == "multiplication":
                    result = X_sample[0] * X_sample[1]
                    details_batch.append(f"{X_sample[0]} * {X_sample[1]} = {result}")
                elif task == "division":
                    result = X_sample[0] // X_sample[1]
                    details_batch.append(f"{X_sample[0]} / {X_sample[1]} = {result}")
                elif task == "exponentiation":
                    result = X_sample[0] ** X_sample[1]
                    details_batch.append(f"{X_sample[0]}^{X_sample[1]} = {result}")
                else:
                    details_batch.append("Unknown task")

            X_batch = np.array(X_batch_list)  # shape: (batch_size, input_size)
            y_batch = np.array(y_batch_list)  # shape: (batch_size, output_size)

            # Forward and backward passes
            output = self.forward(X_batch)
            self.backward_batch(X_batch, y_batch, output)

            # Logging and visualization at specified intervals
            if epoch % vis_interval == 0:
                loss = np.mean((output - y_batch) ** 2)
                print(f"\nEpoch {epoch}/{epochs}, Batch Loss: {loss}")
                print("Example batch problem:", details_batch[0])
                self.visualize_learning_space()

# -----------------------------
# One-Hot Encoding Utility
# -----------------------------
def create_one_hot_encoding(result, max_value):
    """
    Create a one-hot encoded vector for an integer result in [1, max_value].
    """
    one_hot = np.zeros(max_value)
    if 1 <= result <= max_value:
        one_hot[result - 1] = 1
    else:
        raise ValueError(f"Result {result} out of bounds for one-hot encoding with size {max_value}")
    return one_hot

# -----------------------------
# Dataset Generator
# -----------------------------
def generate_math_dataset(num_samples_per_task=1000, output_size=400):
    """
    Generate samples for six tasks with doubled digits for arithmetic problems.
    
    For addition, subtraction, multiplication, division, and algebra:
      - Use numbers in the range 2–20 (i.e. roughly double the original 1–10 range).
    For exponentiation:
      - Use a in [2, 6] and b in [0, 3] (kept modest).
      
    Each sample is a tuple: (input_vector, task, one_hot_encoded_answer)
      * For arithmetic tasks the input is [a, b, 0] (padded to length 3).
      * For algebra the input is [a, b, c] representing: a*x + b = c.
      * For exponentiation the input is [a, b, 0] with result = a**b.
    """
    samples = []

    # Addition: result = a + b
    for _ in range(num_samples_per_task):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        result = a + b
        label = create_one_hot_encoding(result, max_value=output_size)
        samples.append((np.array([a, b, 0]), "addition", label))

    # Subtraction: ensure a >= b
    for _ in range(num_samples_per_task):
        a = random.randint(2, 20)
        b = random.randint(2, a)  # ensure a >= b
        result = a - b
        # Adding 1 to map a result of 0 to index 0 if needed
        label = create_one_hot_encoding(result + 1, max_value=output_size)
        samples.append((np.array([a, b, 0]), "subtraction", label))

    # Multiplication: result = a * b
    for _ in range(num_samples_per_task):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        result = a * b
        label = create_one_hot_encoding(result, max_value=output_size)
        samples.append((np.array([a, b, 0]), "multiplication", label))

    # Division: generate pairs so that a is divisible by b
    for _ in range(num_samples_per_task):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        while a % b != 0:  # Ensure a is divisible by b
            b = random.randint(2, 20)
        result = a // b
        label = create_one_hot_encoding(result, max_value=output_size)
        samples.append((np.array([a, b, 0]), "division", label))

    # Exponentiation: result = a ** b
    for _ in range(num_samples_per_task):
        a = random.randint(2, 6)
        b = random.randint(0, 3)
        result = a ** b
        label = create_one_hot_encoding(result, max_value=output_size)
        samples.append((np.array([a, b, 0]), "exponentiation", label))

    return samples

# -----------------------------
# Run the Neural Network
# -----------------------------
# Hyperparameters
input_size = 3
hidden_size = 16  # for visualization purposes
output_size = 400  # Max value for one-hot encoded output

# Generate dataset
samples = generate_math_dataset(num_samples_per_task=1000, output_size=output_size)

# Instantiate and train the network
network = NeuralNetwork(input_size, hidden_size, output_size, colormap="plasma")
network.train(samples, epochs=100000, batch_size=32, vis_interval=1000)
