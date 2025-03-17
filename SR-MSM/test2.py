import torch

# 获取 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 检查是否支持 CUDA
print(f"CUDA available: {torch.cuda.is_available()}")

# 获取 CUDA 版本
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Device {i} compute capability: {torch.cuda.get_device_capability(i)}")
else:
    print("CUDA is not available.")
