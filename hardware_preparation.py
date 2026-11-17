import torch

def gpu_preparation():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        torch.set_default_device("cuda")
        print("PyTorch default device set to CUDA(GPU)")
        return True
    else:
        print("WARNING: No GPU detected. Performance will be slow")
        return False
    
gpu_preparation()