import torch
import time

compute_device = 'cuda'

# Create a large tensor
size = 10000
tensor = torch.randn(size, size, device=compute_device)


N_RUNS = 10000

# Measure time for slicing along rows
start_time = time.time()
for i in range(N_RUNS):
    idx = torch.randint(0, size-1, (1,1)).item()
    row_slice_result = tensor[idx, :]
row_slice_time = time.time() - start_time

# Measure time for slicing along columns
start_time = time.time()
for i in range(N_RUNS):
    idx = torch.randint(0, size-1, (1,1)).item()
    col_slice_result = tensor[:, idx]
col_slice_time = time.time() - start_time

print(f"Slicing along rows took: {row_slice_time} seconds")
print(f"Slicing along columns took: {col_slice_time} seconds")
