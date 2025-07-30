#!/bin/bash

# install_nemo_simple.sh - 简化版NeMo安装脚本（不依赖conda）
echo "🚀 Installing NeMo Framework (simple version)..."

# 检查Python版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected, will use CPU"
fi

# 安装PyTorch
echo "🧠 Installing PyTorch..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖
echo "📚 Installing base dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install transformers==4.54.0 accelerate==0.34.2 huggingface-hub==0.34.2

# 安装NeMo框架
echo "🤖 Installing NeMo Framework..."
python3 -m pip install "nemo_toolkit[all]"

# 安装其他必要依赖
echo "🔧 Installing additional dependencies..."
python3 -m pip install omegaconf hydra-core
python3 -m pip install tqdm pandas numpy

# 验证安装
echo "✅ Verifying installation..."
python3 -c "
try:
    import nemo
    print('✅ NeMo imported successfully')
except ImportError as e:
    print(f'❌ NeMo import failed: {e}')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('✅ Transformers imported successfully')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} imported successfully')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  CUDA not available')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
"

echo "🎉 NeMo Framework installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Login to HuggingFace: huggingface-cli login"
echo "2. Accept the model license: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm"
echo "3. Run the model: python3 steerlm_nemo_fixed.py"
echo ""
echo "🔗 Useful links:"
echo "- Model page: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm"
echo "- NeMo docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html" 