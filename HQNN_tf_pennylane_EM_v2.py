import os, time, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pennylane as qml
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────────────────────
# Setup
tf.get_logger().setLevel("ERROR")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

try:
    from keras_flops import get_flops
except:
    get_flops = None

def compute_flops_tfprof(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    concrete = tf.function(model).get_concrete_function(tf.TensorSpec([1,28,28,1], tf.float32))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(graph=graph, options=opts)
        return prof.total_float_ops

# ──────────────────────────────────────────────────────────────────────────────
# Quantum Circuit Setup
n_qubits = 4
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")  # batch=False to avoid gradient error
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

class QuantumWrapper(layers.Layer):
    def __init__(self, qlayer, **kwargs):
        super().__init__(**kwargs)
        self.qlayer = qlayer

    def call(self, inputs):
        def single(x):
            x64 = tf.cast(x, tf.float64)
            y64 = self.qlayer(x64)
            return tf.cast(y64, tf.float32)
        return tf.map_fn(single, inputs, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return tf.TensorShape([batch_size, n_qubits])


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Model Definition
def create_hybrid_model():
    inp = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)             # (None, 32)
    x = layers.Dense(2**n_qubits, activation="tanh")(x)
    x = QuantumWrapper(quantum_layer)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inp, out)

# ──────────────────────────────────────────────────────────────────────────────
# Data Prep
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[..., None].astype("float32") / 255.0
X_test = X_test[..., None].astype("float32") / 255.0

# ──────────────────────────────────────────────────────────────────────────────
# Compile & Train
model = create_hybrid_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

os.makedirs("model_figures", exist_ok=True)
os.makedirs("model_metrics_json", exist_ok=True)
os.makedirs("model_figures_all_classes", exist_ok=True)

start_time = time.time()
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, verbose=1)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f} s")

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
t1 = time.time(); _ = model.predict(X_test[:100], verbose=0)
latency = (time.time() - t1) / 100
loss, acc = model.evaluate(X_test, y_test, verbose=0)
params = model.count_params()
flops = get_flops(model, batch_size=1) if get_flops else compute_flops_tfprof(model)

cnn_layers = [l.name for l in model.layers if isinstance(l, layers.Conv2D)]
vqc_layers = [l.name for l in model.layers if "KerasLayer" in type(l).__name__]

probs = model.predict(X_test, verbose=0)
preds = np.argmax(probs, axis=1)
class_acc = {str(i): float((preds[y_test == i] == i).mean()) for i in range(10)}

metrics = {
    "training_time_sec": round(train_time, 2),
    "avg_inference_latency_sec": round(latency, 6),
    "test_accuracy": round(float(acc), 4),
    "total_trainable_params": params,
    "flops_batch1": int(flops),
    "cnn_conv_layers": cnn_layers,
    "vqc_layers": vqc_layers,
    "per_class_accuracy": class_acc
}

print(json.dumps(metrics, indent=2))
with open("model_metrics_json/hqnn_pennylane_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# Plot Training / Validation Accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("HQNN Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.savefig("model_figures/hqnn_train_val_acc.jpg", dpi=300, bbox_inches="tight")
plt.show(); plt.close()

# Plot Per-Class Accuracy
plt.figure(figsize=(6, 4))
plt.bar(range(10), [class_acc[str(i)] for i in range(10)])
plt.xticks(range(10)); plt.ylim(0, 1)
plt.title("Per‑Class Accuracy")
plt.xlabel("Digit"); plt.ylabel("Accuracy")
plt.savefig("model_figures_all_classes/hqnn_per_class_acc.jpg", dpi=300, bbox_inches="tight")
plt.show(); plt.close()
