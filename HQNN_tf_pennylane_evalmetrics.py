import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pennylane as qml
from tensorflow.keras import layers, models

# Try to import keras_flops for FLOPs measurement, but catch any error
try:
    from keras_flops import get_flops
except Exception:
    get_flops = None

# Silence TensorFlow warnings
tf.get_logger().setLevel("ERROR")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Helper: Compute FLOPs via TF Profiler ---
def compute_flops_tfprof(model):
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

    concrete = tf.function(model).get_concrete_function(
        tf.TensorSpec([1, 28, 28, 1], dtype=tf.float32)
    )
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(graph=graph, options=opts)
        return prof.total_float_ops

# --- Quantum Circuit Setup ---
n_qubits = 4
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", batch=True)
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.KerasLayer(
    quantum_circuit, weight_shapes, output_dim=n_qubits
)

# --- Hybrid Model Definition ---
def create_hybrid_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(n_qubits, activation='tanh')(x)
    x = quantum_layer(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- Data Preparation ---
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0
X_train = X_train[..., None]
X_test  = X_test[..., None]

# --- Compile the Model ---
model = create_hybrid_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Ensure output directories exist
os.makedirs("model_figures", exist_ok=True)
os.makedirs("model_metrics_json", exist_ok=True)
os.makedirs("model_figures_all_classes", exist_ok=True)

# --- 1) TRAIN & TIME ---
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.1,
    verbose=1
)
training_time = time.time() - start_time
print(f"Total Training Time: {training_time:.2f} s")

# --- 2) INFERENCE LATENCY ---
num_samples = 100
start_infer = time.time()
_ = model.predict(X_test[:num_samples], verbose=0)
avg_latency = (time.time() - start_infer) / num_samples
print(f"Avg Inference Latency: {avg_latency:.6f} s/sample")

# --- 3) TEST ACCURACY ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# --- 4) PARAMETER COUNT ---
total_params = model.count_params()
print(f"Total Trainable Params: {total_params}")

# --- 5) FLOPs (keras_flops or TF Profiler) ---
if get_flops:
    try:
        flops = get_flops(model, batch_size=1)
    except Exception:
        flops = compute_flops_tfprof(model)
else:
    flops = compute_flops_tfprof(model)
print(f"FLOPs (batch=1): {flops}")

# --- 6) LAYER TYPES ---
cnn_layers = [l.name for l in model.layers if isinstance(l, layers.Conv2D)]
vqc_layers = [l.name for l in model.layers if "KerasLayer" in type(l).__name__]

# --- 7) PER‑CLASS ACCURACY ---
y_probs = model.predict(X_test, verbose=0)
y_preds = np.argmax(y_probs, axis=1)
class_acc = {
    str(i): float((y_preds[y_test == i] == i).mean())
    for i in range(10)
}

# --- 8) AGGREGATE & DUMP METRICS ---
metrics = {
    "training_time_sec": round(training_time, 2),
    "avg_inference_latency_sec": round(avg_latency, 6),
    "test_accuracy": round(float(test_acc), 4),
    "total_trainable_params": total_params,
    "flops_batch1": int(flops),
    "cnn_conv_layers": cnn_layers,
    "vqc_layers": vqc_layers,
    "per_class_accuracy": class_acc
}

print(json.dumps(metrics, indent=2))
with open("model_metrics_json/hqnn_pennylane_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# --- 9) PLOT & SAVE TRAIN/VAL ACCURACY ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Hybrid PennyLane Model: Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.savefig(
    "model_figures/hqnn_pennylane_train_test.jpg",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# --- 10) PLOT & SAVE PER‑CLASS ACCURACY ---
plt.figure(figsize=(6, 4))
plt.bar(range(10), [class_acc[str(i)] for i in range(10)])
plt.xticks(range(10))
plt.ylim(0, 1)
plt.title("Hybrid PennyLane Model: Per‑Class Accuracy")
plt.xlabel("Digit")
plt.ylabel("Accuracy")
plt.savefig(
    "model_figures_all_classes/hqnn_pennylane_per_class.jpg",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
