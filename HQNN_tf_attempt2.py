import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Quantum Circuit Setup ---
# Use 4 qubits and 3 variational layers
n_qubits = 4
n_layers = 3

# Create a PennyLane device; "default.qubit" is a noiseless simulator.
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", batch=True)
def quantum_circuit(inputs, weights):
    # Here, inputs is assumed to have shape (batch_size, n_qubits)
    # Use AngleEmbedding to map classical features to quantum states.
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    # Use StronglyEntanglingLayers as the variational ansatz.
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Return expectation values for each qubit.
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define the weight shapes required by StronglyEntanglingLayers.
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Create a Keras-compatible quantum layer.
quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

# --- Hybrid Model Definition ---
def create_hybrid_model():
    # Set input shape to full MNIST image dimensions (28x28, 1 channel).
    inputs = tf.keras.Input(shape=(28, 28, 1))
    
    # A classical convolutional block for feature extraction.
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    # Map the classical features to a vector of length equal to the number of qubits.
    x = layers.Dense(n_qubits, activation='tanh')(x)
    
    # Insert the quantum layer.
    x = quantum_layer(x)
    
    # Final classification layer: change from 2 outputs (binary) to 10 outputs (10 classes).
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- Data Preparation ---
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Use the original MNIST data without filtering and with full resolution.
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0
X_train = X_train[..., None]
X_test  = X_test[..., None]

# --- Model Compilation, Training, and Evaluation ---
model = create_hybrid_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train the model.
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)

# Evaluate on the test set.
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Plot training history.
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Hybrid Quantum–Classical Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()
