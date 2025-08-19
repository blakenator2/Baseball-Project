import numpy as np
import pickle
import copy
import pandas as pd
import sys
sys.setrecursionlimit(10000)

class Layer:
    def __init__(self, n_inputs, n_neurons,
        weight_regularizer_l1=0, weight_regularizer_l2=0,
        bias_regularizer_l1=0, bias_regularizer_l2=0):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1, self.weight_regularizer_l2  = weight_regularizer_l1, weight_regularizer_l2
        self.bias_regularizer_l1, self.bias_regularizer_l2 = bias_regularizer_l1, bias_regularizer_l2

    def forward(self, inputs, training): 
        self.inputs = inputs  # Remember input values

        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        # Ensure dvalues is a NumPy array
        if isinstance(dvalues, (pd.DataFrame, pd.Series)):
            dvalues = dvalues.to_numpy()

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 
    
        # Gradients on regularization
        if self.weight_regularizer_l1 > 0: # L1 on weights
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
            
        if self.weight_regularizer_l2 > 0: # L2 on weights
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
            
        if self.bias_regularizer_l1 > 0: # L1 on biases
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
            
        if self.bias_regularizer_l2 > 0: # L2 on biases
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
         
        self.dinputs = np.dot(dvalues, self.weights.T) #Gradient on values

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Dropout:

    def __init__(self, rate):
    # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs # Save input values

    # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask # Apply mask to output values
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask  # Gradient on values

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

class ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs # Remember input values
        self.output = np.maximum(0, inputs)# Calculate output values from inputs

    def backward(self, dvalues):
    # Since we need to modify original variable, let's make a copy of values first
        self.dinputs = dvalues.copy()
    
        self.dinputs[self.inputs <= 0] = 0 # Zero gradient where input values were negative
    
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
    
class Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs # Remember input values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) # Get unnormalized probabilities
         
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True) # Normalize them for each sample
        self.output = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1) # Flatten output array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) # Calculate Jacobian matrix of the output
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues) # Calculate sample-wise gradient and add it to the array of sample gradients
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    
class Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs # Save input and calculate
        self.output = 1 / (1 + np.exp(-inputs)) # Save output of the sigmoid function

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output # Derivative - calculates from output of the sigmoid function

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs # Just remember values

    def backward(self, dvalues):
        self.dinputs = dvalues.copy() # derivative is 1, 1 * dvalues = dvalues - the chain rule

    def predictions(self, outputs):
        return outputs

class SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.): # learning rate of 1. is default for this optimizer
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights) # If layer does not contain momentum arrays, create them filled with zeros
                layer.bias_momentums = np.zeros_like(layer.biases) # If there is no momentum array for weights the array doesn't exist for biases yet either.
                
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates # Build weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients
            
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates # Build bias updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            
            layer.weights += weight_updates
            layer.biases += bias_updates # Update weights and biases using either vanilla or momentum updates
    
    def post_update_params(self):
        self.iterations += 1

class Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):  # If layer does not contain cache arrays, create them filled with zeros
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2 # Update cache with squared current gradients

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon) # Vanilla SGD parameter update + normalization with square rooted cache
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'): # If layer does not contain cache arrays, create them filled with zeros
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2 # Update cache with squared current gradients
        
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon) # Vanilla SGD parameter update + normalization with square rooted cache
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):# If layer does not contain cache arrays, create them filled with zeros
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 *  layer.bias_momentums + (1 - self.beta_1) * layer.dbiases # Update momentum with current gradients

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums /  (1 - self.beta_1 ** (self.iterations + 1)) # Get corrected momentum self.iteration is 0 at first pass and we need to start with 1 here

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2 # Update cache with squared current gradients
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1)) # Get corrected cache

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_params(self):
         self.iterations += 1

class Loss:
    def regularization_loss(self):
        regularization_loss = 0
        
        for layer in self.trainable_layers: # Calculate regularization loss iterate all trainable layers
            if layer.weight_regularizer_l1 > 0: # L1 regularization - weights calculate only when factor greater than 0
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            
            if layer.weight_regularizer_l2 > 0: # L2 regularization - weights
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
            if layer.bias_regularizer_l1 > 0: # L1 regularization - biases calculate only when factor greater than 0
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0: # L2 regularization - biases
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y) # Calculate sample losses
        data_loss = np.mean(sample_losses) # Calculate mean loss
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization: # If just data loss - return it
            return data_loss
        
        return data_loss, self.regularization_loss() # Return the data and regularization losses
     
    def new_pass(self): # Reset variables for accumulated loss
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count # Calculate mean loss
        if not include_regularization: # If just data loss - return it
            return data_loss

        return data_loss, self.regularization_loss()

class Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred) # Number of samples in a batch

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip data to prevent division by 0 and to not drag mean towards any value
        
        if len(y_true.shape) == 1: # Probabilities for target values - only if categorical labels
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        elif len(y_true.shape) == 2: # Mask values - only for one-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidences) # Losses
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues) # Number of samples
        labels = len(dvalues[0]) # Number of labels in every sample, we'll use the first sample to count them
        
        if len(y_true.shape) == 1: # If labels are sparse, turn them into one-hot vector
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues # Calculate gradient 
        self.dinputs = self.dinputs / samples # Normalize gradient

class Softmax_Entropy():
    def backward(self, dvalues, y_true):
        samples = len(dvalues)# Number of samples
        
        if len(y_true.shape) == 2: # If labels are one-hot encoded, turn them into discrete values
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy() # Copy so we can safely modify
        
        self.dinputs[range(samples), y_true] -= 1 # Calculate gradient
        self.dinputs = self.dinputs / samples # Normalize gradient

class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clip data to prevent division by 0 and to not drag mean towards any value

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)) # Calculate sample-wise loss
        sample_losses = np.mean(sample_losses, axis=-1)
        
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues) # Number of samples
        outputs = len(dvalues[0]) # Number of outputs in every sample, we'll use the first sample to count them
        
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7) # Clip data to prevent division by and to not drag mean towards any value
        
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs # Calculate gradient
        self.dinputs = self.dinputs / samples # Normalize gradient

class MeanSquaredError(Loss): # L2 loss
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=1)  # Calculate loss. 1 for pd datafram, -1 for np array
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues) # Number of samples
        outputs = len(dvalues[0]) # Number of outputs in every sample, we'll use the first sample to count them
        self.dinputs = -2 * (y_true - dvalues) / outputs # Gradient on values
        self.dinputs = self.dinputs / samples # Normalize gradient

class MeanAbsoluteError(Loss): # L1 loss
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)# Calculate loss
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues) # Number of samples
        outputs = len(dvalues[0]) # Number of outputs in every sample, we'll use the first sample to count them
        self.dinputs = np.sign(y_true - dvalues) / outputs # Calculate gradient
        self.dinputs = self.dinputs / samples # Normalize gradient

class Accuracy:
    def calculate(self, predictions, y): # Calculates an accuracy given predictions and ground truth values
        comparisons = self.compare(predictions, y) # Get comparison results
        accuracy = np.mean(comparisons) # Calculate an accuracy
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count # Calculate an accuracy
        return accuracy

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y): # No initialization is needed
        pass

    def compare(self, predictions, y): # Compares predictions to the ground truth values
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None # Create precision property

    def init(self, y, reinit=False): # Calculates precision value based on passed-in ground truth values
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    def compare(self, predictions, y): # Compares predictions to the ground truth values
        return np.absolute(predictions - y) < self.precision

class Model:
    def __init__(self):
        self.layers = [] # Create a list of network objects
        self.softmax_classifier_output = None # Softmax classifier's output object

    def add(self, layer): # Add objects to the model
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None): # Set loss, optimizer and accuracy
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input() # Create and set the input layer
        layer_count = len(self.layers) # Count all the objects
        self.trainable_layers = [] # Initialize a list containing trainable layers

        for i in range(layer_count):
            if i == 0: # If it's the first layer, the previous layer object is the input layer
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:  #All layers except for the first and the last
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else: # The last layer - the next object is the loss
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss # Let's save aside the reference to the last object whose output is the model's output
                self.output_layer_activation = self.layers[i]


            if hasattr(self.layers[i], 'weights'): # If layer contains an attribute called "weights", it's a trainable layer 
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None: # Update loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy create an object of combined activation
        # and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, Entropy):
            self.softmax_classifier_output = Softmax_Entropy() # Create an object of combined activation and loss functions

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y) # Initialize accuracy object
        
        train_steps = 1 # Default value if batch size is not being set

        
        if batch_size is not None: # Calculate number of steps
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

        
        for epoch in range(1, epochs+1):# Main training loop
            print(f'epoch: {epoch}') # Print epoch number
            self.loss.new_pass()
            self.accuracy.new_pass() # Reset accumulated values in loss and accuracy objects

            for step in range(train_steps): # Iterate over steps
                if batch_size is None: # If batch size is not set - train using one step and full dataset
                    batch_X = X
                    batch_y = y
                else: # Otherwise slice a batch
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions,batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params() # Optimize (update parameters)
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f} (' + f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' + f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated() # Get and print epoch loss and accuracy

            print(f'training, ' + f'acc: {epoch_accuracy:.3f}, ' + f'loss: {epoch_loss:.3f} (' + f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' + f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None: # If there is the validation data
                self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1# Default value if batch size is not being set

        if batch_size is not None: # Calculate number of steps
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass() # Reset accumulated values in loss and accuracy objects
        
        for step in range(validation_steps): # Iterate over steps
            if batch_size is None: # If batch size is not set - train using one step and full dataset
                batch_X = X_val
                batch_y = y_val
            else: # Otherwise slice a batch
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X, training=False)

            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()


        print(f'validation, ' + f'acc: {validation_accuracy:.3f}, ' + f'loss: {validation_loss:.3f}')

    def forward(self, X, training):
        # Call forward method on the input layer this will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain and pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output # "layer" is now the last object from the list, return its output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None: # If softmax classifier
            # First call backward method on the combined activation/loss this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer which is Softmax activation as we used combined activation/loss object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through all the objects but last in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        
        # This will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def get_parameters(self):
        parameters = [] # Create a list for parameters
        for layer in self.trainable_layers: # Iterable trainable layers and get their parameters
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        # Iterate over the parameters and layers and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def save_parameters(self, path):
        with open(path, 'wb') as file: # Open a file in the binary-write mode and save parameters to it
            pickle.dump(self.get_parameters(), file)
    
    def load_parameters(self, path):
        with open(path, 'rb') as file: # Open a file in the binary-write mode and save parameters to it
            self.set_parameters(pickle.load(file))

    def save(self, path):
        model = copy.deepcopy(self)  # Make a deep copy of current model instance
        
        model.loss.new_pass()
        model.accuracy.new_pass() # Reset accumulated values in loss and accuracy objects
        
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None) # Remove data from the input layer and gradients from the loss object
        
        for layer in model.layers:# For each layer remove inputs, output and dinputs properties
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        with open(path, 'wb') as f:# Open a file in the binary-write mode and save the model
            pickle.dump(model, f)
    
    @staticmethod #allows you to call model without previously initializing it
    def load(path):
        with open(path, 'rb') as file: # Open file in the binary-read mode, load a model
            model = pickle.load(file)
        return model

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1 # Default value if batch size is not being set
        output = [] # Model outputs
        
        if batch_size is not None:# Calculate number of steps
            prediction_steps = len(X) // batch_size
        
            if prediction_steps * batch_size < len(X): # Dividing rounds down. If there are some remaining data, but not a full batch, this won't include it
                prediction_steps += 1 # Add `1` to include this not full batch
        
        for step in range(prediction_steps): # Iterate over steps
        
            if batch_size is None: # If batch size is not set - train using one step and full dataset
                batch_X = X
        
            else: # Otherwise slice a batch
                batch_X = X[step*batch_size:(step+1)*batch_size]

            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output) # Append batch prediction to the list of predictions
     
        return np.vstack(output)