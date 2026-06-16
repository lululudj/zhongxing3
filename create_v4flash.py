"""Ollama创建v4-flash模型"""
import urllib.request, json, os, sys

OLLAMA_BASE = "http://127.0.0.1:11434"
GGUF_PATH = "E:/models/v4-flash-download/deepseek-v4-flash-iq2xxs.gguf"

if not os.path.exists(GGUF_PATH):
    print(f"文件不存在: {GGUF_PATH}")
    sys.exit(1)

size_gb = os.path.getsize(GGUF_PATH) / 1e9
print(f"GGUF: {size_gb:.1f} GB")

modelfile = f'''FROM {GGUF_PATH}
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
'''

print(f"Modelfile:\n{modelfile}")

payload = json.dumps({
    "name": "v4-flash",
    "modelfile": modelfile,
    "stream": False
}).encode()

req = urllib.request.Request(
    f"{OLLAMA_BASE}/api/create",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print("创建 v4-flash...")
try:
    resp = urllib.request.urlopen(req, timeout=600)
    result = json.loads(resp.read())
    if result.get("status") == "success":
        print("创建成功!")
    else:
        print(f"结果: {result}")
except Exception as e:
    print(f"失败: {e}")
    sys.exit(1)

print("\n模型列表:")
req2 = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
models = json.loads(urllib.request.urlopen(req2).read())
for m in models.get("models", []):
    sz = m['size'] / 1e9
    print(f"  {m['name']:30s} {sz:.1f} GB")
