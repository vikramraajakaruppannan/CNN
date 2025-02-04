import numpy as np
import struct

# Function to load MNIST image files (without .gz)
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 28, 28)
        return images / 255.0  # Normalize to [0,1]

# Function to load MNIST label files (without .gz)
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load dataset from non-gzipped files
x_train = load_mnist_images("mnist1/train-images-idx3-ubyte/train-images-idx3-ubyte")
y_train = load_mnist_labels("mnist1/train-labels-idx1-ubyte/train-labels-idx1-ubyte")
x_test = load_mnist_images("mnist1/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
y_test = load_mnist_labels("mnist1/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("MNIST dataset loaded successfully!")


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

# Save the trained model
model.save("mnist_cnn.h5")
print("Model training complete and saved successfully!")
