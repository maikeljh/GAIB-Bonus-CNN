import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Global predict
predict = -1

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# MSE Loss Function
def mean_square_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Softmax Activation Function
def softmax(x):
    # Subtract the maximum value of x to prevent overflow
    exp_x = np.exp(x)
    
    # Compute softmax values
    softmax_output = exp_x / np.sum(exp_x, axis=0)

    return softmax_output

# Abstract Base Class for layers
class Layer:
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, d_output):
        raise NotImplementedError
    
    def update_weights(self, learning_rate):
        raise NotImplementedError

# Dense layer class
class Dense(Layer):
    def __init__(self, input_size, output_size, activation_function):
        # Constructor
        super().__init__()

        # Data dimension
        self.input_size = input_size

        # Units
        self.output_size = output_size

        # Activation Function
        self.activation_function = activation_function

        # Initialize weights and biases with random values
        self.weights = np.random.randn(output_size, input_size) / input_size
        self.biases = np.zeros(output_size)
        
        # Gradients for weight and bias updates
        self.d_weights = np.zeros((output_size, input_size))
        self.d_biases = np.zeros(output_size)

    def forward(self, inputs):
        # Save input data
        self.inputs = inputs

        # Compute pre activation values of the neurons
        self.z = np.dot(inputs, self.weights.T) + self.biases

        # Applying activation function to each neuron values
        self.output = self.activation_function(self.z)

        return self.output

    def backward(self, d_output):
        global predict

        # Calculate the gradient of the loss with respect to the pre-activation values
        d_activation = d_output[predict] * self.activation_derivative(self.z)

        # Calculate the gradient of the loss with respect to inputs of layer
        d_input = self.weights.T @ d_activation

        # Calculate the gradient of the loss with respect to the weights of the layer
        self.d_weights = self.inputs[np.newaxis].T @ d_activation[np.newaxis]
        self.d_weights = self.d_weights.T

        # Calculate the gradient of the loss with respect to the biases of the layer
        self.d_biases = d_activation

        return d_input

    def activation_derivative(self, x):
        # For this assignment, only implemented for sigmoid and relu
        if self.activation_function == sigmoid:
            # Define derivative of sigmoid
            return sigmoid(x) * (1 - sigmoid(x))
        elif self.activation_function == relu:
            # Define derivative of relu
            return (x > 0).astype(float)
        elif self.activation_function == softmax:
            # Define derivative of softmax
            global predict

            # Calculate the exponential transformation of the input array (x) and sum of all
            exp = np.exp(x)
            exp_total = np.sum(exp)

            # Compute gradients with respect to output (Z) for target index
            result = -exp[predict] * exp / (exp_total ** 2)
            result[predict] = exp[predict] * (exp_total - exp[predict]) / (exp_total ** 2)

            return result
        else:
            raise NotImplementedError("Activation function derivative not implemented.")

    def update_weights(self, learning_rate):
        # Stochastic Gradient Descent
        # Substract weights with product of learning rate and derivative weights
        self.weights -= learning_rate * self.d_weights

        # Substract biases with product of learning rate and derivative weights
        self.biases -= learning_rate * self.d_biases

# Sequential model class
class Sequential:
    def __init__(self):
        # Constructor
        self.layers = []

    def add(self, layer):
        # Add new layer
        self.layers.append(layer)

    def forward(self, inputs):
        # Iterate all layers, connecting neurons from layers to layers
        for layer in self.layers:
            # Pass output from current layer to the next layer
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, d_output):
        # Iterate all layers in reverse order
        for layer in reversed(self.layers):
            # Calculate gradients for current layer and pass to next layer
            d_output = layer.backward(d_output)

    def update_weights(self, learning_rate):
        # Iterate all layers
        for layer in self.layers:
            # Update weights using gradients computed and scales them with learning rate
            layer.update_weights(learning_rate)

    def fit(self, X, y, epochs=100, learning_rate=0.01, batch_size=32):
        # Train ANN
        for epoch in range(epochs):
            # Define total loss and total accuracy
            total_loss = 0
            total_correct = 0

            # Define size of data
            num_samples = len(X)

            # Shuffle the data for each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            # Progress Bar
            progress_bar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for batch_start in progress_bar:
                # Create a batch
                batch_indices = indices[batch_start: batch_start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward and backward propagation for the current batch
                # Define batch loss and accuracy
                batch_loss = 0
                batch_correct = 0

                # Iterate for all batch data
                for i in range(len(X_batch)):
                    # Define inputs
                    inputs = X_batch[i]

                    # Define target
                    target = y_batch[i]

                    # Forward propagation
                    prediction = self.forward(inputs)

                    # Compute loss
                    loss = mean_square_error(target, prediction)
                    batch_loss += loss

                    # Calculate accuracy
                    if np.argmax(target) == np.argmax(prediction):
                        batch_correct += 1

                    # Backward propagation
                    global predict
                    predict = np.argmax(target)

                    d_output = np.zeros(len(prediction))
                    d_output[predict] = -1 / prediction[predict]

                    self.backward(d_output)

                    # Weight update using SGD
                    self.update_weights(learning_rate)

                # Calculate average loss and average accuracy for the batch
                average_loss = batch_loss / len(X_batch)
                average_accuracy = batch_correct / len(X_batch)

                # Save total loss and total correct
                total_loss += average_loss
                total_correct += average_accuracy

                # Update progress bar
                progress_bar.set_postfix(loss=average_loss, accuracy=average_accuracy)

            # Calculate average loss for the epoch
            average_loss = total_loss / (num_samples // batch_size)
            average_accuracy = total_correct / (num_samples // batch_size)

            # Print final result for current epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}")

    def predict(self, X):
        # Define predictions
        predictions = []

        # Iterate all test data
        for i in range(len(X)):
            # Get features (inputs)
            inputs = X[i]

            # Predict
            prediction = self.forward(inputs)
            
            # Make prediction for multi class classification
            prediction = np.argmax(prediction)

            # Add prediction
            predictions.append(prediction)

        # Return predictions
        return np.array(predictions)

    def save_model(self, file_path):
        # Serialize and save the model using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    def load_model(file_path):
        # Deserialize and load the model using pickle
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    def display_image(X, y):
        # Get the image
        image = X

        # Display the image using matplotlib
        plt.imshow(image, cmap='gray')
        plt.title(f"Image - Label: {y}")
        plt.axis('off')
        plt.show()

# CNN
# Convolutional Layer class
class Conv2D(Layer):
    def __init__(self, filters, kernel_size, input_channels):
        # Constructor
        # Define number of filters, size of kernel, and number of input channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels

        # Generate random filters of shape and weight normalization
        self.kernels = np.random.randn(filters, kernel_size, kernel_size, input_channels) / (kernel_size ** 2)

    def patches_generator(self, image):
        # Divide the input image in patches to be used during convolution.
        # Extract image height and width
        image_h, image_w, _ = image.shape

        # Save image
        self.image = image

        # The number of patches, given a kernel_size * kernel_size filter 
        # is image_h - kernel_size + 1 for height and image_w - kernel_size + 1 for width
        for h in range(image_h - self.kernel_size + 1):
            for w in range(image_w - self.kernel_size + 1):
                patch = image[h : (h + self.kernel_size), w : (w + self.kernel_size), :]
                yield patch, h, w
    
    def forward(self, image):
        # Extract image height and width
        image_h, image_w, _ = image.shape

        # Initialize the convolution output volume of the correct size
        convolution_output = np.zeros((image_h - self.kernel_size + 1, image_w - self.kernel_size + 1, self.filters))

        # Unpack the patches generator
        for patch, h, w in self.patches_generator(image):
            # Perform convolution for each patch
            conv = np.sum(patch * self.kernels, axis=(1, 2, 3))
            convolution_output[h, w] = conv

        # Return output
        return convolution_output

    def backward(self, dE_dY):
        # Initialize gradient of the loss function with respect to the kernel weights
        self.dE_dk = np.zeros(self.kernels.shape)

        # Initialize backward output
        self.out = np.zeros(self.image.shape)

        # Iterate through image patches and compute gradients for each kernel
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.filters):
                # Accumulate gradients for the kernel weights using the chain rule
                self.dE_dk[f] += patch[:, :, :] * dE_dY[h, w, f]

                # Accumulate gradients for the output image using the chain rule
                self.out[h : h + self.kernel_size, w : w + self.kernel_size] += dE_dY[h, w, f] * self.kernels[f]

        return self.out
    
    def update_weights(self, learning_rate):
        # Stochastic Gradient Descent
        # Subtract weights with the product of the learning rate and derivative kernels
        self.kernels -= learning_rate * self.dE_dk

# MaxPool2D Layer class
class MaxPool2D(Layer):
    def __init__(self, pool_size):
        # Constructor
        super().__init__()

        # Set pool size
        self.pool_size = pool_size

    def patches_generator(self, image):
        # Compute the ouput size
        output_h = image.shape[0] // self.pool_size
        output_w = image.shape[1] // self.pool_size

        # Save image
        self.image = image

        # Iterate making all patches based on height and width
        for h in range(output_h):
            for w in range(output_w):
                # Extract a patch from the image using the pool_size
                patch = image[(h * self.pool_size):(h * self.pool_size + self.pool_size),
                              (w * self.pool_size):(w * self.pool_size + self.pool_size)]
                yield patch, h, w

    def forward(self, image):
        # Extract image height, width, and number of filters
        image_h, image_w, num_kernels = image.shape

        # Initialize max pooling output
        max_pooling_output = np.zeros((image_h // self.pool_size, image_w // self.pool_size, num_kernels))

        # Iterate through image patches and perform max pooling operation
        for patch, h, w in self.patches_generator(image):
             # Find the maximum value along the spatial dimensions (axis 0 and axis 1) for each kernel
            max_pooling_output[h,w] = np.amax(patch, axis=(0,1))

        return max_pooling_output

    def backward(self, dE_dY):
        # Initialize the gradient of the loss function with respect to the input image
        dE_dk = np.zeros(self.image.shape)

        # Iterate through image patches
        for patch, h, w in self.patches_generator(self.image):
            # Extract image height, width, and number of filters
            image_h, image_w, num_kernels = patch.shape

            # Find the maximum value along the spatial dimensions (axis 0 and axis 1) for each kernel in the current patch
            max_val = np.amax(patch, axis=(0,1))

            # Iterate through all elements
            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        # Check if the value in the current position is the maximum value in the corresponding kernel
                        if patch[idx_h, idx_w, idx_k] == max_val[idx_k]:
                            # Propagate the gradients from the output to the location of the maximum value in the original image
                            dE_dk[h * self.pool_size + idx_h, w * self.pool_size + idx_w, idx_k] = dE_dY[h, w, idx_k]

        return dE_dk
    
    def update_weights(self, learning_rate):
        # Stochastic Gradient Descent
        # Do nothing
        return

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        # Store the original shape of the input image for later use in backward propagation
        self.original_shape = image.shape

        # Flatten the input image into a 1D vector
        flattened_output = image.flatten()

        return flattened_output

    def backward(self, dE_dY):
        # Reshape the gradient to match the original shape of the input image
        dE_dX = dE_dY.reshape(self.original_shape)
        return dE_dX

    def update_weights(self, learning_rate):
        # Flatten layer does not have any weights to update, so this method does nothing
        return