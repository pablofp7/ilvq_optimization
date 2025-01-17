import torch
import torch.nn as nn
import torch.quantization
from collections import deque

class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_units=[32, 16], output_size=1, quantize=False):
        super(DynamicMLP, self).__init__()
        
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], output_size),
            nn.Sigmoid()
        )
        self.gradient_queue = deque(maxlen=10)  
        
        
        if quantize:
            self.quantize_model()

    def forward(self, x):
        return self.layers(x)

    def quantize_model(self):
        """Apply dynamic quantization to the model."""
        self.layers = torch.quantization.quantize_dynamic(
            self.layers,  
            {nn.Linear},  
            dtype=torch.qint8  
        )

    def compute_gradients(self, x_batch, y_batch, criterion):
        """Compute gradients for a batch of data."""
        self.zero_grad()
        outputs = self(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        gradients = [param.grad.clone() for param in self.parameters()]
        return gradients

    def apply_gradients(self, gradients, optimizer):
        """Apply aggregated gradients to update the model."""
        for param, grad in zip(self.parameters(), gradients):
            param.grad = grad
        optimizer.step()

    def aggregate_gradients(self, local_gradients, neighbor_gradients):
        """Aggregate local gradients with gradients from neighbors."""
        aggregated_gradients = []
        for local_grad, *neighbor_grads in zip(local_gradients, *neighbor_gradients):
            avg_grad = torch.mean(torch.stack([local_grad] + list(neighbor_grads)), dim=0)
            aggregated_gradients.append(avg_grad)
        return aggregated_gradients

    def process_gradient_queue(self, optimizer, criterion):
        """Process gradients from the queue during idle time."""
        while self.gradient_queue:
            neighbor_gradients = self.gradient_queue.pop()  
            local_gradients = self.compute_gradients(
                torch.rand(1, 5),
                torch.randint(2, size=(1, 1), dtype=torch.float32),
                criterion
            )
            aggregated_gradients = self.aggregate_gradients(local_gradients, [neighbor_gradients])
            self.apply_gradients(aggregated_gradients, optimizer)
