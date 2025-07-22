# ------------------ Imports ------------------
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, concatenate
from tensorflow.keras.utils import plot_model

# ------------------ Load & Preprocess MNIST ------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# ------------------ Display a Sample Image ------------------
plt.imshow(x_train[5], cmap='gray')
plt.title(f"Image Label = {np.argmax(y_train[5])}")
plt.axis('off')
plt.show()

# ------------------ Show Unique Labels ------------------
print("Unique Labels:", np.unique(np.argmax(y_train, axis=1)))

# ------------------ Sequential Model ------------------
seq_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(5, activation='relu'),
    Dense(10, activation='softmax')
])

seq_model.summary()

seq_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

seq_model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)

# Evaluate on test set
loss, accuracy = seq_model.evaluate(x_test, y_test)
print(f"\nSequential Model Test Accuracy: {accuracy:.4f}")

# Optionally, view weights
# print(seq_model.get_weights())

# ------------------ Functional API Model ------------------
input_layer = Input(shape=(28, 28))
flatten = Flatten()(input_layer)

# Two branches from the same flattened input
hidden1 = Dense(128, activation='relu')(flatten)
hidden2 = Dense(256, activation='relu')(flatten)

# Further layer from one branch
hiddenl1 = Dense(64, activation='relu')(hidden1)

# Merge the two branches
merged = concatenate([hiddenl1, hidden2])

# Output layer
output_layer = Dense(10, activation='softmax')(merged)

# Build model
func_model = Model(inputs=input_layer, outputs=output_layer)
func_model.summary()

# Plot model architecture to file
plot_model(func_model, show_shapes=True, show_layer_names=True, to_file='functional_model.png')

# Train functional model
func_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
func_model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)
func_model.evaluate(x_test, y_test)

