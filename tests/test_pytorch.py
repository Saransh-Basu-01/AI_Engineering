import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

x = torch.rand(3, 3)
print(f"Random tensor:\n{x}")
print("✅ PyTorch is working")