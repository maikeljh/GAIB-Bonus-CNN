# Michael Jonathan Halim 13521124
# GAIB - Bonus - CNN

# Load libraries
import numpy as np
import pandas as pd
from cnn import Sequential, Conv2D, MaxPool2D, Flatten, Dense, softmax, relu, sigmoid
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess the data
# Load the data (replace data with your data filename)
data = pd.read_csv('./test/data.csv')

# Extract features (pixel values) and labels
# Adjust how many data you want to train and test
X = data.iloc[:6000, 1:].values.astype(np.float32)
y = data.iloc[:6000, 0].values.astype(int)

# Current preprocessing image based on 28x28x1 MNIST dataset
# Normalize the pixel values to range [0, 1]
X /= 255.0

# Reshape X to match the expected input shape for the Conv2D layer
X = X.reshape(-1, 28, 28, 1)

# Split to train and test set
# Change test size and random state
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

# Show example image
Sequential.display_image(X_train[0], y_train[0])

# Create the CNN model (You can create your own model)
# My model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_channels=1))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(input_size=13*13*16, output_size=10, activation_function=softmax))

# (LeNet), appreantly accuracy is not great
# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=5, input_channels=1))
# model.add(MaxPool2D(pool_size=2))
# model.add(Conv2D(filters=16, kernel_size=5, input_channels=6))
# model.add(MaxPool2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(input_size=4*4*16, output_size=120, activation_function=relu))
# model.add(Dense(input_size=120, output_size=84, activation_function=relu))
# model.add(Dense(input_size=84, output_size=10, activation_function=softmax))

# Train the model
# Convert labels to one-hot encoding
from keras.utils import to_categorical

# For this dataset, it has 10 classes
y_one_hot = to_categorical(y_train, num_classes=10)

# Load model (optional)
# Change the relative path to your relative path
# model = Sequential.load_model('./test/model.pkl')

# Train the model
# Adjust epochs, batch size, and learning rate
model.fit(X_train, y_one_hot, epochs=10, batch_size=32, learning_rate=0.05)

# Test the model
y_pred = model.predict(X_test)

# Evaluate
classification_rep = classification_report(y_test, y_pred)

# Print Classification Report
print("\nClassification Report:")
print(classification_rep)

# Save model (optional)
# Change the relative path to your relative path
# model.save_model('./test/model.pkl')