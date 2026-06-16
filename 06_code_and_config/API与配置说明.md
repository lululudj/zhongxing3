# API 与配置说明

> **更新 [2026-06-14]**：zhongxing_agent.py 已从 `requests` 库迁移为 `urllib` 标准库实现，不再需要 `pip install requests`。

## Ollama API 调用

### 基础调用

```python
def call_ollama(model: str, prompt: str, system: str = "", timeout: int = 180) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": msgs, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]
```

### 特殊模型处理

| 模型 | 处理 |
|------|------|
| deepseek-r1 | 拼接 system+prompt，去掉 `</think` 标记 |
| minicpm | 在 prompt 后添加 `/no_think` |
| qwen3.5:4b | 需 `/no_think` 关闭 thinking 模式 |

### JSON 解析

```python
def try_parse_json(raw: str):
    # 尝试：直接解析 → 提取 [] 内容 → 提取 {} 内容
```

### 错误处理

- ConnectionError → 提示用户启动 Ollama
- 超时 → 返回 ERROR 字符串
- 异常 → 返回错误信息