import sys
import io

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print("✅ GPU DETECTED: " + torch.cuda.get_device_name(0))
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("❌ NO GPU DETECTED (or CUDA not installed).")
        print("Training will be extremely slow or impossible.")
except ImportError:
    print("❌ PyTorch not installed. Cannot check GPU.")
