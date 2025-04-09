import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Import Qiskit components.
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer  # Proper import for Aer

# --- Quantum Circuit Function using Qiskit ---
def run_quantum_circuit(input_vector, weights):
    """
    Constructs a parameterized quantum circuit in Qiskit,
    encodes input data via rotation gates, then applies trainable
    rotation gates (with entanglement) and simulates the circuit to 
    compute the expectation value of Z for each qubit.
    
    Args:
        input_vector (np.array): Array of shape (n_qubits,) encoding the input.
        weights (np.array): Trainable parameters of shape (n_layers, n_qubits, 3).
    
    Returns:
        np.array: Expectation values for each qubit; shape (n_qubits,).
    """
    # Ensure inputs are proper numpy arrays of floats.
    input_vector = np.array(input_vector, dtype=float)
    weights = np.array(weights, dtype=float)
    
    n_qubits = len(input_vector)
    circuit = QuantumCircuit(n_qubits)
    
    # Encode input: apply RY rotations.
    for i in range(n_qubits):
        # Convert to python float explicitly.
        circuit.ry(float(input_vector[i]), i)
        
    # Determine the number of layers.
    n_layers = weights.shape[0]
    
    # Apply trainable rotations and a chain of CNOTs for entanglement.
    for l in range(n_layers):
        for q in range(n_qubits):
            circuit.ry(float(weights[l, q, 0]), q)
            circuit.rz(float(weights[l, q, 1]), q)
            circuit.rx(float(weights[l, q, 2]), q)
        # Apply entanglement: a chain of CNOT gates.
        for q in range(n_qubits - 1):
            circuit.cx(q, q+1)
    
    # Use the Aer statevector simulator.
    backend = Aer.get_backend("statevector_simulator")
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit)
    result = job.result()
    state = result.get_statevector(transpiled_circuit)
    
    # Compute expectation values for the Pauli-Z operator on each qubit.
    expec = np.zeros(n_qubits, dtype=np.float32)
    n_states = len(state)
    for i in range(n_qubits):
        prob0 = 0.0
        for j in range(n_states):
            # Check if bit corresponding to qubit i is 0 in the state index.
            if ((j >> (n_qubits - i - 1)) & 1) == 0:
                prob0 += np.abs(state[j])**2
        expec[i] = 2 * prob0 - 1  # <Z> = 2*P(0) - 1.
    return expec

# --- Custom Gradient Wrapper for the Quantum Simulation ---
@tf.custom_gradient
def quantum_forward_with_grad(x, w):
    """
    Wraps the quantum simulation in a custom gradient.
    (For demonstration purposes, this returns zero gradients so the quantum parameters are frozen.)
    """
    # Use tf.py_function and let TensorFlow convert the tensors to numpy arrays.
    y = tf.py_function(
            func=lambda x_val, w_val: run_quantum_circuit(x_val, w_val),
            inp=[x, w],
            Tout=tf.float32)
    # Set the output shape explicitly (we assume it matches the shape of x).
    y.set_shape(x.shape)
    
    def grad(dy):
        # Return zero gradients for both x and w.
        return tf.zeros_like(x), tf.zeros_like(w)
    return y, grad

# --- Custom Keras Layer using Qiskit ---
class QiskitQuantumLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that wraps a Qiskit quantum circuit.
    It stores trainable parameters and, for each input sample,
    applies the quantum simulation using the custom gradient wrapper.
    """
    def __init__(self, n_layers, n_qubits, **kwargs):
        super(QiskitQuantumLayer, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        
    def build(self, input_shape):
        # Create trainable weights for the quantum circuit.
        self.weights_var = self.add_weight(
            name="weights",
            shape=(self.n_layers, self.n_qubits, 3),
            initializer="random_normal",
            trainable=True
        )
        super(QiskitQuantumLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Applies the quantum simulation to each input sample.
        Each sample is assumed to be of shape (n_qubits,).
        """
        def quantum_forward(x):
            return quantum_forward_with_grad(x, self.weights_var)
        
        outputs = tf.map_fn(quantum_forward, inputs, dtype=tf.float32)
        outputs.set_shape((inputs.shape[0], self.n_qubits))
        return outputs
    
    def get_config(self):
        config = super(QiskitQuantumLayer, self).get_config()
        config.update({
            "n_layers": self.n_layers,
            "n_qubits": self.n_qubits,
        })
        return config

# --- Hybrid Model Definition ---
def create_hybrid_model_qiskit():
    """
    Constructs a hybrid model that uses classical convolutional layers to extract features 
    from MNIST images, reduces the features to match the number of qubits,
    passes them through a Qiskit quantum layer, and finally classifies the output.
    """
    # Input: 28x28 grayscale images.
    inputs = tf.keras.Input(shape=(28, 28, 1))
    
    # Classical CNN layers.
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    
    # Reduce feature dimensionality to match the number of qubits.
    n_qubits = 4  # Adjust as needed.
    x = layers.Dense(n_qubits, activation="tanh")(x)
    
    # Apply the custom Qiskit quantum layer.
    n_layers = 2  # Number of quantum circuit layers.
    x = QiskitQuantumLayer(n_layers, n_qubits)(x)
    
    # Final classification layer for 10 classes.
    outputs = layers.Dense(10, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# --- Data Preparation: Load MNIST ---
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = X_train[..., None]  # Add channel dimension.
X_test = X_test[..., None]

# --- Compile, Train, and Evaluate the Model ---
model = create_hybrid_model_qiskit()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# --- Plot Training History ---
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Hybrid Qiskit Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
