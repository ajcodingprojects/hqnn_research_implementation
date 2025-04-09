import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Quantum Circuit Setup ---
# For binary classification on low-resolution images, we use 4 qubits.
n_qubits = 4
# Increase the number of variational layers to improve circuit expressibility.
n_layers = 3

# Create a PennyLane device; default.qubit is a noiseless simulator.
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", batch=True)
def quantum_circuit(inputs, weights):
    # Here, inputs is assumed to have shape (batch_size, n_qubits)
    # Use AngleEmbedding for feature mapping.
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    # Use StronglyEntanglingLayers as variational ansatz.
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Return the expectation values for each qubit.
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define the weight shapes. StronglyEntanglingLayers needs weights of shape (n_layers, n_qubits, 3).
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Create a Keras-compatible quantum layer.
quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

# --- Hybrid Model Definition ---
def create_hybrid_model():
    # Input shape set to match downsampled images (here, 4x4 grayscale).
    inputs = tf.keras.Input(shape=(4, 4, 1))
    
    # A deeper classical block for robust feature extraction.
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    # Add a second convolutional block.
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    # Add an extra Dense layer to boost feature extraction.
    x = layers.Dense(32, activation='relu')(x)
    # Map the features to a vector of size n_qubits.
    x = layers.Dense(n_qubits, activation='tanh')(x)
    
    # Quantum layer.
    x = quantum_layer(x)
    
    # Final classification layer for binary classification (digits 0 and 1)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# --- Data Preparation ---
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Filter dataset to only include digits 0 and 1.
train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter  = np.where((y_test == 0) | (y_test == 1))
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test   = X_test[test_filter], y_test[test_filter]

# Downsample images to 4x4 to match the quantum layer capacity.
# Using TensorFlow's image resize.
X_train = tf.image.resize(X_train[..., None], (4, 4)).numpy()
X_test  = tf.image.resize(X_test[..., None], (4, 4)).numpy()

# Normalize pixel values to [0, 1].
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# --- Model Compilation, Training, and Evaluation ---
model = create_hybrid_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train for an extended number of epochs to help the model converge.
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Hybrid Quantum–Classical Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()
