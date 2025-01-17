import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from collections import deque
import pickle

class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_units=[32, 16], output_size=1, quantize=True, learning_rate=0.001):
        """
        Initializes the DynamicMLP model.
        
        Arguments:
        - input_size: Number of features in the input data.
        - hidden_units: List specifying the number of neurons in hidden layers.
        - output_size: Number of output neurons (1 for binary classification).
        - quantize: If True, apply dynamic quantization to reduce the model's size.
        - learning_rate: Learning rate for the optimizer.
        """
        super(DynamicMLP, self).__init__()
        
        # Define the neural network layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], output_size),
            nn.Sigmoid()
        )
        
        # Queue to store received gradients from neighbors
        self.gradient_queue = deque(maxlen=10)
        
        # Apply quantization
        if quantize:
            self.quantize_model()

        # Initialize loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def quantize_model(self):
        """
        Apply dynamic quantization to reduce the model's size and improve efficiency.
        """
        self.layers = torch.quantization.quantize_dynamic(
            self.layers,  
            {nn.Linear},  
            dtype=torch.qint8  
        )

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
            sample: A single input sample (as a tensor).
        
        Returns:
            The model's prediction (as a tensor).
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            sample = sample.unsqueeze(0)  # Add batch dimension
            prediction = self.layers(sample)
        return prediction
    
    def learn_one(self, sample, label):
        """
        Trains the model on a single input sample and its corresponding label.
        
        Args:
            sample: A single input sample (as a tensor).
            label: The corresponding label (as a tensor).
        
        Returns:
            The loss value (as a tensor).
        """
        self.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Zero the gradients
        sample = sample.unsqueeze(0)  # Add batch dimension
        output = self.layers(sample)  # Forward pass
        loss = self.criterion(output, label.unsqueeze(0))  # Compute loss
        loss.backward()  # Backward pass
        self.optimizer.step()  # Update weights
        return loss.item()
    
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
                aggregated_params[key] += torch.tensor(params[key]) * weight

        # Update the local model's parameters with the aggregated result
        self.load_state_dict(aggregated_params)
