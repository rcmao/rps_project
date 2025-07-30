#!/bin/bash

# install_nemo_china_mirror.sh - 使用国内镜像安装NeMo框架
echo "🚀 Installing NeMo Framework with Chinese mirrors..."

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

# 配置国内镜像
echo "🌐 Configuring Chinese mirrors..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
retries = 3
EOF

# 设置环境变量
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export PIP_TIMEOUT=120
export PIP_RETRIES=3

echo "✅ Pip mirror configured: https://pypi.tuna.tsinghua.edu.cn/simple/"

# 升级pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 安装PyTorch（使用清华镜像）
echo "🧠 Installing PyTorch from Tsinghua mirror..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖（使用清华镜像）
echo "📚 Installing base dependencies from Tsinghua mirror..."
python3 -m pip install transformers==4.54.0 accelerate==0.34.2 huggingface-hub==0.34.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 安装NeMo框架（使用清华镜像）
echo "🤖 Installing NeMo Framework from Tsinghua mirror..."
python3 -m pip install "nemo_toolkit[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 安装其他必要依赖（使用清华镜像）
echo "🔧 Installing additional dependencies from Tsinghua mirror..."
python3 -m pip install omegaconf hydra-core -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3 -m pip install tqdm pandas numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 配置HuggingFace镜像
echo "🌐 Configuring HuggingFace mirror..."
mkdir -p ~/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export TRANSFORMERS_CACHE=/root/.cache/huggingface

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

echo "🎉 NeMo Framework installation completed with Chinese mirrors!"
echo ""
echo "📋 Next steps:"
echo "1. Login to HuggingFace: huggingface-cli login"
echo "2. Accept the model license: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm"
echo "3. Run the model: python3 steerlm_nemo_fixed.py"
echo ""
echo "🔗 Useful links:"
echo "- Model page: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm"
echo "- NeMo docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html"
echo ""
echo "🌐 Mirror configuration:"
echo "- Pip: https://pypi.tuna.tsinghua.edu.cn/simple/"
echo "- HuggingFace: https://hf-mirror.com" 