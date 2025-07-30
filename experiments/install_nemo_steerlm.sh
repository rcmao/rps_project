#!/bin/bash

# install_nemo_steerlm.sh - 正确安装NeMo框架和SteerLM模型
echo "🚀 Installing NeMo Framework for SteerLM models..."

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

# 创建conda环境（如果不存在）
if ! conda env list | grep -q "nemo"; then
    echo "📦 Creating conda environment 'nemo'..."
    conda create --name nemo python=3.10.12 -y
fi

# 激活环境
echo "🔧 Activating nemo environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nemo

# 安装PyTorch
echo "🧠 Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装基础依赖
echo "📚 Installing base dependencies..."
pip install --upgrade pip
pip install transformers==4.54.0 accelerate==0.34.2 huggingface-hub==0.34.2

# 安装NeMo框架
echo "🤖 Installing NeMo Framework..."
pip install "nemo_toolkit[all]"

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
"

echo "🎉 NeMo Framework installation completed!"
echo "💡 To use: conda activate nemo" 