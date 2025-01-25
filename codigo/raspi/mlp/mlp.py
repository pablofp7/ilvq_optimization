import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from collections import deque

class DynamicMLP(nn.Module):
    def __init__(
        self, 
        input_size,
        hidden_units=[32, 16],
        output_size=1,
        quantize=True,
        learning_rate=0.01,
        dropout_rate=0.1,
        weight_decay=0.1,
        ):
        
        """
        Initializes the DynamicMLP model.

        Args:
            input_size: Number of input features.
            hidden_units: List specifying the number of neurons in each hidden layer.
            output_size: Number of output neurons (1 for binary classification).
            quantize: If True, apply dynamic quantization.
            learning_rate: Learning rate for the optimizer.
            dropout_rate: Dropout rate to use between layers (0.0 means no dropout).
            weight_decay: L2 regularization strength (0.0 means no regularization).
        """
        super(DynamicMLP, self).__init__()

        # Dynamically create layers
        layers = []
        prev_units = input_size
        for units in hidden_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_units = units

        # Add the output layer
        layers.append(nn.Linear(prev_units, output_size))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

        # Apply quantization
        if quantize:
            self.quantize_model()

        # Initialize loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Add L2 regularization

    def quantize_model(self):
        """
        Applies dynamic quantization to the linear layers in the model, reducing memory usage
                and computational requirements.            
        """
        for name, module in self.layers.named_children():
            if isinstance(module, nn.Linear):
                setattr(self.layers, name, torch.quantization.quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8))

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Arguments:
        - x: Input tensor.
        
        Returns:
        - Output tensor after passing through the network.
        """
        return self.layers(x)

    def get_parameters(self):
        """
        Extracts the parameters (weights and biases) of the model.
        
        Returns:
            A dictionary containing the model's parameters.
        """
        return self.state_dict()
    
    def predict_one(self, sample):
        """
        Predicts the output for a single input sample.
        Args:
            sample: A dictionary with feature indices as keys and feature values as values.
        Returns:
            int: The predicted label (0 or 1).
        """
        self.eval()
        with torch.no_grad():
            # Ensure consistent feature order
            feature_order = sorted(sample.keys())
            sample_tensor = torch.tensor([sample[f] for f in feature_order], dtype=torch.float32).unsqueeze(0)
            
            # Print input tensor
            print(f"Input Tensor: {sample_tensor}")
            
            # Apply sigmoid to ensure probability output
            probability = self(sample_tensor)
            print(f"Raw Probability: {probability.item()}")
            
            label = 1 if probability.item() >= 0.5 else 0
            print(f"Predicted Label: {label}")
        return label

    def learn_one(self, sample, label):
        """
        Trains the model on a single input sample and its corresponding label.
        """
        self.train()
        self.optimizer.zero_grad()

        # Forward propagation
        feature_order = sorted(sample.keys())
        sample_tensor = torch.tensor([sample[f] for f in feature_order], dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32).unsqueeze(1)

        # Print input tensor and label
        print(f"Input Tensor: {sample_tensor}")
        print(f"Label Tensor: {label_tensor}")

        # Forward pass
        output = self(sample_tensor)
        print(f"Raw Output: {output.item()}")

        # Compute loss
        loss = self.criterion(output, label_tensor)
        print(f"Loss: {loss.item()}")

        # Backward propagation
        loss.backward()

        # Check gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name} - Grad Norm: {param.grad.norm().item()}")

        # Update weights
        self.optimizer.step()

        return loss.item()

    def learn_batch(self, samples, labels):
        """
        Trains the model on a batch of input samples and their corresponding labels.

        Args:
            samples: A list of dictionaries, where each dictionary represents a sample.
            labels: A list of labels corresponding to the samples.

        Returns:
            The average loss value for the batch.
        """
        self.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Zero the gradients

        # Convert list of dictionaries to a batch tensor
        sample_tensors = [torch.tensor([v for _, v in sorted(sample.items())], dtype=torch.float32) for sample in samples]
        sample_batch = torch.stack(sample_tensors)  # Shape: [batch_size, input_size]

        # Convert labels to a batch tensor
        label_batch = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Shape: [batch_size, 1]

        # Forward pass
        output = self.layers(sample_batch)  # Shape: [batch_size, 1]

        # Compute loss
        loss = self.criterion(output, label_batch)  # Scalar value

        # Backward pass and optimization
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update weights

        return loss.item()

    def learn_sliding_window(self, sample, label, window_size=100):
        """
        Trains the model using a sliding window of recent samples.

        Args:
            sample: A dictionary representing the new sample.
            label: The label corresponding to the new sample.
            window_size: The size of the sliding window (default: 100).

        Returns:
            The average loss value for the window.
        """
        if not hasattr(self, 'sample_buffer'):
            # Initialize the sliding window buffers
            self.sample_buffer = deque(maxlen=window_size)
            self.label_buffer = deque(maxlen=window_size)

        # Add the new sample and label to the buffers
        self.sample_buffer.append(sample)
        self.label_buffer.append(label)

        # Train on the entire window
        loss = self.learn_batch(list(self.sample_buffer), list(self.label_buffer))
        return loss

    def test_then_train(self, sample, label):
        """
        First predicts the output for a single sample (test phase), then trains the model on that sample (train phase).
        
        Args:
            sample: A single input sample (as a tensor).
            label: The corresponding label (as a tensor).
        
        Returns:
            A tuple containing the prediction and the loss value.
        """
        # Test phase: Predict the output
        prediction = self.predict_one(sample)
        
        # Train phase: Train on the sample
        loss = self.learn_one(sample, label)
        
        return prediction, loss
    
    def aggregate_and_update_parameters(self, list_of_params, weights=None):
        """
        Aggregates parameters from neighbors' models (including the local model)
        using weighted averaging and updates the current model's parameters in-place.

        Args:
            list_of_params: A list of dictionaries, where each dictionary contains the parameters of a model.
            weights: A list of weights for each model. If None, uniform weights are used.
        """
        # Include the local model's parameters in the aggregation
        current_params = self.get_parameters()  # Deserialize local parameters
        list_of_params = [current_params] + list_of_params  # Add local parameters to the list

        # If no weights are provided, use uniform weights
        if weights is None:
            weights = [1.0 / len(list_of_params)] * len(list_of_params)

        # Validate weights length
        if len(weights) != len(list_of_params):
            raise ValueError("The length of weights must match the number of parameter sets.")

        # Initialize a dictionary to store the aggregated parameters
        aggregated_params = {key: torch.zeros_like(current_params[key]) for key in current_params.keys()}

        # Perform weighted averaging
        for params, weight in zip(list_of_params, weights):
            for key in params:
                aggregated_params[key] += params[key].clone().detach() * weight  # Fix applied here

        # Update the local model's parameters with the aggregated result
        self.load_state_dict(aggregated_params)