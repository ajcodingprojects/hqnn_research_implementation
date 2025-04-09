import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load and Preprocess the MNIST Dataset
# MNIST dataset: 60,000 training images and 10,000 test images.
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1 by scaling with 1/255.
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the data to include a channel dimension (28, 28, 1) since the images are grayscale.
X_train = X_train[..., None]
X_test = X_test[..., None]

# 2. Define the CNN Architecture
# The network includes:
# - Convolutional layers for feature extraction.
# - Max-pooling layers for downsampling.
# - A Flatten layer to convert 2D data to 1D.
# - Dense (fully-connected) layers for classification.
model = models.Sequential([
    # First convolutional block: 32 filters, a 3x3 kernel and ReLU activation.
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block: 64 filters, a 3x3 kernel and ReLU activation.
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block: 64 filters, a 3x3 kernel.
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten the output from the conv layers to feed into the dense layers.
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    
    # Output layer: 10 units for the 10 digits, using softmax activation for classification.
    layers.Dense(10, activation='softmax')
])

# 3. Compile the Model
# Loss function: sparse_categorical_crossentropy (since labels are integers).
# Optimizer: Adam.
# Metrics: Accuracy.
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Optional: Print the model summary to check the architecture.
model.summary()

# 4. Train the Model
# We train for 5 epochs and reserve 10% of the training data for validation.
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# 5. Evaluate the Model on Test Data
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# 6. Plot Training and Validation Accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()
