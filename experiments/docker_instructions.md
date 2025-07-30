
# 🐳 Docker 使用说明

## 启动 Docker 服务

如果 Docker 服务未运行，请尝试以下命令：

```bash
# 方法1：使用 systemctl
sudo systemctl start docker
sudo systemctl enable docker

# 方法2：使用 service
sudo service docker start

# 方法3：直接启动
sudo dockerd &
```

## 使用官方 NeMo 容器

```bash
# 启动官方 NeMo 容器
docker run --gpus all -it --rm nvcr.io/nvidia/nemo:25.02

# 在容器内运行 Nemotron 模型
python -c "
from nemo.deploy import NemoQuery
nq = NemoQuery(url='localhost:8000', model_name='Nemotron-3-8B-Chat-4K-SteerLM')
# 使用官方 SteerLM 格式
"
```

## 启动 NeMo 推理服务器

```bash
# 启动推理服务器
docker run -d --gpus all -p 8000:8000 nvcr.io/nvidia/nemo:25.02 \
    python -m nemo.deploy.inference.server \
    --model-path /path/to/Nemotron-3-8B-Chat-4k-SteerLM.nemo \
    --port 8000
```

## 检查 Docker 状态

```bash
# 检查 Docker 版本
docker --version

# 检查 Docker 服务状态
docker ps

# 检查 GPU 支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
