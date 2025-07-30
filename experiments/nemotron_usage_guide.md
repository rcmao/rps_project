
# Nemotron-3-8B-SteerLM 使用指南

## 🎯 最佳方案

根据官方文档，推荐使用以下方法：

### 方案1：官方 NeMo Docker 容器（推荐）
```bash
# 启动官方 NeMo 容器
docker run --gpus all -it --rm nvcr.io/nvidia/nemo:25.02

# 在容器内运行
python -c "
from nemo.deploy import NemoQuery
nq = NemoQuery(url='localhost:8000', model_name='Nemotron-3-8B-Chat-4K-SteerLM')
# 使用官方 SteerLM 格式
"
```

### 方案2：NeMo 推理服务器
```bash
# 启动推理服务器
docker run -d --gpus all -p 8000:8000 nvcr.io/nvidia/nemo:25.02 \
    python -m nemo.deploy.inference.server \
    --model-path /path/to/Nemotron-3-8B-Chat-4k-SteerLM.nemo \
    --port 8000
```

## 📝 官方 SteerLM 格式

```python
PROMPT_TEMPLATE = """<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>quality:4,understanding:4,correctness:4,coherence:4,complexity:4,verbosity:4,toxicity:0,humor:0,creativity:0,violence:0,helpfulness:4,not_appropriate:0,hate_speech:0,sexual_content:0,fails_task:0,political_content:0,moral_judgement:0,lang:en"""
```

## 🎛️ 属性控制

每个属性可以设置为 0-4 的值：
- quality: 响应质量
- understanding: 理解程度
- correctness: 正确性
- coherence: 连贯性
- complexity: 复杂度
- verbosity: 详细程度
- toxicity: 毒性
- humor: 幽默感
- creativity: 创造性
- violence: 暴力内容
- helpfulness: 有用性
- not_appropriate: 不当内容
- hate_speech: 仇恨言论
- sexual_content: 性内容
- fails_task: 任务失败
- political_content: 政治内容
- moral_judgement: 道德判断

## 📚 参考资料

- 官方文档: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm
- NeMo 框架: https://github.com/NVIDIA/NeMo
- SteerLM 论文: https://arxiv.org/abs/2310.05344
