import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Quantum Circuit Setup ---

# Set the number of qubits and layers for the quantum circuit.
n_qubits = 4
n_layers = 2

# Create a PennyLane device; "default.qubit" is a noiseless simulator.
dev = qml.device("default.qubit", wires=n_qubits)

# Define a quantum circuit.
# Here we set batch=True so that the QNode accepts a batched input.
@qml.qnode(dev, interface="tf", batch=True)
def quantum_circuit(inputs, weights):
    # Here, inputs is assumed to be a tensor of shape (batch_size, n_qubits)
    # Iterate over each qubit and apply an RY gate using the column corresponding to that qubit.
    for i in range(n_qubits):
        qml.RY(inputs[:, i], wires=i)
    # Use a template for entanglement and trainable parameters.
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Return the expectation values of the Pauli-Z observable for each qubit.
    # With batch=True, each returned value is a tensor of shape (batch_size, 1)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define the shape of the weights needed for the quantum circuit.
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Create a Keras-compatible layer using the quantum circuit.
quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

# --- Hybrid Model Definition ---

def create_hybrid_model():
    # Input layer: MNIST images are 28x28 grayscale.
    inputs = tf.keras.Input(shape=(28, 28, 1))
    
    # Classical convolutional block.
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    # Reduce feature dimensions to match the number of qubits.
    x = layers.Dense(n_qubits, activation='tanh')(x)
    
    # Quantum layer; note that the output dimension is set to n_qubits.
    x = quantum_layer(x)
    
    # Final classification layer: softmax over 10 classes.
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Construct and return the hybrid model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# --- Data Preparation ---

# Load MNIST data.
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize images to the range [0, 1] and add a channel dimension.
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = X_train[..., None]
X_test = X_test[..., None]

# --- Model Compilation, Training, and Evaluation ---

# Create the hybrid model.
model = create_hybrid_model()

# Compile the model using Adam and sparse categorical crossentropy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture.
model.summary()

# Train the model (adjust epochs as needed).
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model on the test set.
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# --- Plot Training History ---

plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Hybrid Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()
