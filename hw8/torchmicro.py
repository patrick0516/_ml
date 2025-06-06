import torch

x = torch.tensor(5.0, requires_grad=True)
y = torch.tensor(5.0, requires_grad=True)
z = torch.tensor(5.0, requires_grad=True)

def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

learning_rate = 0.1
iterations = 100

for i in range(iterations):
    loss = f(x, y, z)
    
    loss.backward()
    
    with torch.no_grad():
        x -= learning_rate * x.grad
        y -= learning_rate * y.grad
        z -= learning_rate * z.grad
    
    x.grad.zero_()
    y.grad.zero_()
    z.grad.zero_()
    
    print(f"Iteration {i+1}: x = {x.item()}, y = {y.item()}, z = {z.item()}, f(x, y, z) = {loss.item()}")
