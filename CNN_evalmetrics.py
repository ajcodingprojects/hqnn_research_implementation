import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

    # Build a concrete function from the Keras model
    concrete = tf.function(model).get_concrete_function(
        tf.TensorSpec([1, 28, 28, 1], dtype=tf.float32)
    )
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

    # Import the graph and run the profiler
    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(graph=graph, options=opts)
        return prof.total_float_ops

# 1. Load and Preprocess the MNIST Dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0
X_train = X_train[..., None]
X_test  = X_test[..., None]

# 2. Define the CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compile the Model
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

# 4. Train the Model (and time it)
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.1,
    verbose=1
)
training_time = time.time() - start_time
print(f"Total Training Time: {training_time:.2f} s")

# 5. Measure Inference Latency
num_samples = 100
start_infer = time.time()
_ = model.predict(X_test[:num_samples], verbose=0)
avg_latency = (time.time() - start_infer) / num_samples
print(f"Avg Inference Latency: {avg_latency:.6f} s/sample")

# 6. Evaluate Test Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 7. Parameter Count
total_params = model.count_params()
print(f"Total Trainable Params: {total_params}")

# 8. FLOPs (try keras_flops, else TF Profiler fallback)
if get_flops:
    try:
        flops = get_flops(model, batch_size=1)
    except Exception:
        flops = compute_flops_tfprof(model)
else:
    flops = compute_flops_tfprof(model)
print(f"FLOPs (batch=1): {flops}")

# 9. Layer Types
cnn_layers = [l.name for l in model.layers if isinstance(l, layers.Conv2D)]
vqc_layers = []  # Not applicable for a pure CNN

# 10. Per‑Class Accuracy
y_probs = model.predict(X_test, verbose=0)
y_preds = np.argmax(y_probs, axis=1)
class_acc = {
    str(i): float((y_preds[y_test == i] == i).mean())
    for i in range(10)
}

# 11. Aggregate & Dump Metrics to JSON
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
with open("model_metrics_json/cnn_metrics_more_epochs.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 12. Plot & Save Training/Validation Accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Classical CNN: Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.savefig(
    "model_figures/cnn_train_test_more_epochs.jpg",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# 13. Plot & Save Per‑Class Accuracy Distribution
plt.figure(figsize=(6, 4))
plt.bar(range(10), [class_acc[str(i)] for i in range(10)])
plt.xticks(range(10))
plt.ylim(0, 1)
plt.title("Classical CNN: Per‑Class Accuracy")
plt.xlabel("Digit")
plt.ylabel("Accuracy")
plt.savefig(
    "model_figures_all_classes/cnn_per_class_more_epochs.jpg",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
