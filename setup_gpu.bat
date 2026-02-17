@echo off
echo --- SEPHLIGHTY AI GPU SETUP ---
echo This script installs the software needed to run your AI on a GPU.
echo Requirements: NVIDIA GPU (RTX 3060 or better) + CUDA Toolkit 12.1

echo.
echo 1. Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 2. Installing Unsloth (Training Tool)...
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

echo.
echo 3. Verifying Installation...
python backend/verify_gpu.py

pause
