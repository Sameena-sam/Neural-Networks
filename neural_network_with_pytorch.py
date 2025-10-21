import torch
import torch.nn as nn
import torch.optim as optim

# Input and target
X = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
y = torch.tensor([[1]], dtype=torch.float32)

# Model
model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Forward → Backward → Update
output = model(X)
loss = criterion(output, y)
loss.backward()
optimizer.step()

# Show result
print("Predicted Output:", output.detach().numpy())
print("Loss:", loss.item())
















































