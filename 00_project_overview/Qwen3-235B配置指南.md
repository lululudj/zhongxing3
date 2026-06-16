# Qwen3-235B-A22B MoE 混合推理配置

## 硬件配置
- GPU: RTX 3060 8GB / RTX 4060 8GB
- CPU: i5-10400F (6核12线程) / i7-13620H (10核16线程)
- RAM: 32GB DDR4/DDR5
- SSD: NVMe 1TB+

## 模型信息
- 模型: Qwen3-235B-A22B-Instruct-2507-IQ2_M.gguf
- 大小: 78.5GB
- 总参数: 235B
- 激活参数: 22B
- 专家数: 128个，每次激活8个

## llama.cpp 启动参数

### RTX 3060 8GB 配置
```bash
llama-server.exe \
  -m "E:\models\Qwen3-235B\Qwen_Qwen3-235B-A22B-Instruct-2507-IQ2_M.gguf" \
  -ngl 10 \                    # GPU层数（Attention层）
  --n-cpu-moe 999 \            # MoE专家卸载到CPU
  --flash-attn on \            # Flash Attention
  -c 4096 \                    # 上下文长度
  -t 6 \                       # CPU线程数
  -b 512 \                     # 批处理大小
  --cache-type-k q4_0 \        # KV cache量化
  --cache-type-v q4_0 \
  --mlock \                    # 锁定内存
  --host 127.0.0.1 \
  --port 8080
```

### RTX 4060 8GB 配置（开发机）
```bash
llama-server.exe \
  -m "E:\models\Qwen3-235B\Qwen_Qwen3-235B-A22B-Instruct-2507-IQ2_M.gguf" \
  -ngl 15 \                    # GPU层数（更多层）
  --n-cpu-moe 999 \            # MoE专家卸载到CPU
  --flash-attn on \
  -c 8192 \                    # 更大上下文
  -t 10 \                      # 更多CPU线程
  -b 512 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --mlock \
  --host 127.0.0.1 \
  --port 8080
```

## 内存分配策略

| 组件 | 位置 | 大小 |
|------|------|------|
| Attention层 | GPU | ~3GB |
| 共享专家 | GPU | ~2GB |
| KV Cache | GPU | ~1GB |
| 激活的8个专家 | CPU RAM | ~15GB |
| 专家缓存 | CPU RAM | ~10GB |
| 模型文件 | SSD mmap | 78.5GB |

## 预期性能

| 指标 | RTX 3060 | RTX 4060 |
|------|----------|----------|
| 首token延迟 | 5-10s | 3-5s |
| 生成速度 | 2-5 tok/s | 5-10 tok/s |
| 单次回答时间 | 10-20s | 5-15s |

## 众星管道集成

```python
# zhongxing_moe_235b.py
def call_qwen235b_brain(prompt: str, context: str = "") -> str:
    """调用 Qwen3-235B 作为大脑"""
    import urllib.request, json

    body = {
        "model": "qwen3-235b",
        "messages": [
            {"role": "system", "content": f"机语:\n{context}"},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        }
    }

    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=300)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]
```

## 下载模型

```bash
# 使用 huggingface_hub
pip install huggingface_hub
python download_qwen.py
```

## 验证安装

```bash
# 测试模型
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-235b", "messages": [{"role": "user", "content": "你好"}]}'
```
