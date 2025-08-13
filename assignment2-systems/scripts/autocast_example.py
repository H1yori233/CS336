import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"Input dtype: {x.dtype}")
        x = self.relu(self.fc1(x))
        print(f"After fc1 and relu dtype: {x.dtype}")
        x = self.ln(x)
        print(f"After ln dtype: {x.dtype}")
        x = self.fc2(x)
        print(f"After fc2 (output) dtype: {x.dtype}")
        return x


# Ensure CUDA is available
if torch.cuda.is_available():
    device = "cuda"
    model = ToyModel(10, 10).to(device)
    dtype: torch.dtype = torch.float16

    # Input tensor x is created with the target dtype
    x: torch.Tensor = torch.randn(10, 10, device=device, dtype=torch.float32)

    print("--- Running with autocast ---")
    # Autocast context manager
    with torch.autocast(device_type=device, dtype=dtype):
        # The input tensor is cast to float16 inside the model's first autocast-eligible op
        # if it's not already float16.
        # Let's pass in a float32 tensor to observe autocast's behavior.
        y = model(x)

    print("\n--- Final Output Tensor Properties ---")
    print(f"Final y dtype: {y.dtype}")
    print(f"Final y device: {y.device}")
else:
    print("CUDA device not found. This example requires a GPU.")
