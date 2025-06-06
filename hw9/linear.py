import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)  
X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)  
y = 2 * X + 1 + 0.1 * torch.randn(X.size())  

plt.scatter(X.numpy(), y.numpy())
plt.xlabel('X')
plt.ylabel('y')
plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)  

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Estimated weight: {model.linear.weight.item():.4f}, bias: {model.linear.bias.item():.4f}')

predicted = model(X).detach()
plt.scatter(X.numpy(), y.numpy(), label='Original data')
plt.plot(X.numpy(), predicted.numpy(), label='Fitted line', color='r')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
