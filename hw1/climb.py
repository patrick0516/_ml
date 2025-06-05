x, y, z = 0.0, 0.0, 0.0 
alpha = 0.1          
for i in range(100):
    grad_x = 2*x - 2
    grad_y = 2*y - 4
    grad_z = 2*z - 6

    x = x - alpha * grad_x
    y = y - alpha * grad_y
    z = z - alpha * grad_z

print(f"最小值點為: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
print(f"對應的函數值為: f = {x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8:.4f}")
