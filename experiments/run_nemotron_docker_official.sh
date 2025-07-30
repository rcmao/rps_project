#!/bin/bash

# 启动NeMo推理服务器
echo "🚀 Starting NeMo inference server..."
docker run -d --name nemotron-server \
    --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8000:8000 \
    nvcr.io/nvidia/nemo:25.02 \
    python -m nemo.deploy.inference.server \
    --model-path /workspace/Nemotron-3-8B-Chat-4k-SteerLM.nemo \
    --port 8000

# 等待服务器启动
echo "⏳ Waiting for server to start..."
sleep 30

# 运行测试
echo "🧪 Running test..."
python3 load_nemotron_official.py

# 清理
echo "🧹 Cleaning up..."
docker stop nemotron-server
docker rm nemotron-server

echo "✅ Docker solution completed!"
