import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# --- Quantum Circuit Setup ---
# Specify the number of qubits and layers.
n_qubits = 4
n_layers = 2

# Create a PennyLane device using a noiseless simulator.
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit as a QNode.
@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    # Encode classical inputs into quantum amplitudes with RY rotations.
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    # Apply a template that uses trainable parameters and entanglement.
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Return the expectation value of Pauli-Z for each qubit.
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Custom Quantum Layer ---
# This custom Keras layer wraps the quantum circuit, bypassing the KerasLayer version check.
class CustomQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_qubits, **kwargs):
        super(CustomQuantumLayer, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.n_qubits = n_qubits

    def build(self, input_shape):
        # Create a trainable weight variable for the quantum circuit.
        self.weights_var = self.add_weight(
            name="weights",
            shape=(self.n_layers, self.n_qubits, 3),
            initializer="random_normal",
            trainable=True,
        )
        super(CustomQuantumLayer, self).build(input_shape)

    def call(self, inputs):
        # 'inputs' is expected to have shape (batch_size, n_qubits).
        # Define a function that applies the quantum circuit to one input sample.
        def circuit_forward(x):
            return quantum_circuit(x, self.weights_var)
        # Use tf.map_fn to apply the circuit to each sample in the batch.
        outputs = tf.map_fn(circuit_forward, inputs, dtype=tf.float32)
        return outputs

    def get_config(self):
        config = super(CustomQuantumLayer, self).get_config()
        config.update({
            "n_layers": self.n_layers,
            "n_qubits": self.n_qubits,
        })
        return config

# --- Hybrid Model Definition ---
def create_hybrid_model():
    # Input: MNIST images (28 x 28 x 1).
    inputs = tf.keras.Input(shape=(28, 28, 1))
    
    # Classical convolutional blocks.
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    
    # Reduce the dimensionality to match the number of qubits.
    x = layers.Dense(n_qubits, activation="tanh")(x)
    
    # Quantum layer: use the custom quantum layer defined above.
    x = CustomQuantumLayer(n_layers, n_qubits)(x)
    
    # Final dense layer: output probability distribution over 10 classes.
    outputs = layers.Dense(10, activation="softmax")(x)
    
    # Build and return the model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# --- Data Preparation ---
# Load MNIST data.
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize images to [0, 1] and add a channel dimension.
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = X_train[..., None]
X_test = X_test[..., None]

# --- Model Compilation, Training, and Evaluation ---
model = create_hybrid_model()

model.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

model.summary()

# Train the model.
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the hybrid model on the test set.
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# --- Plot Training History ---
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
