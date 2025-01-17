import torch
import torch.nn as nn
import torch.optim as optim
from mlp import DynamicMLP

input_size = 5  # Number of features
model = DynamicMLP(input_size=input_size)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simulate real sample arrivals and idle time
for epoch in range(10):
    # Simulate real sample arrival
    x_real = torch.rand(1, input_size)
    y_real = torch.randint(2, size=(1, 1), dtype=torch.float32)

    # Compute local gradients
    local_gradients = model.compute_gradients(x_real, y_real, criterion)

    # Simulate receiving gradients from neighbors (add to queue)
    neighbor_gradients = [torch.rand_like(grad) for grad in local_gradients]  # Simulated neighbor gradients
    model.gradient_queue.append(neighbor_gradients)
    
    print(f"Queue before processing: {model.gradient_queue}") if epoch == 0 else None
    # Process gradient queue during idle time
    model.process_gradient_queue(optimizer, criterion)

    print(f"Epoch {epoch+1}, Queue Size: {len(model.gradient_queue)}")