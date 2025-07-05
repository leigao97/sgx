import torch

# Adjust path as needed
lib = torch.ops.load_library("build/libops_sim.so")

x = torch.randn(3, 4)
print(x)
y = torch.ops.ops_sim.add_one(x)
print(y)  # x + 1
z = torch.ops.ops_sim.add_noise(x)
print(z)  # x + noise