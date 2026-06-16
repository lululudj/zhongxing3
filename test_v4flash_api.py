"""测试llama-server V4-Flash API，输出到stderr"""
import urllib.request
import json
import time
import sys

start = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Sending request...", file=sys.stderr, flush=True)

data = json.dumps({
    "model": "deepseek-v4-flash-iq2xxs",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 5,
    "temperature": 0
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)

try:
    r = urllib.request.urlopen(req, timeout=3600)
    elapsed = time.time() - start
    resp = json.loads(r.read())
    content = resp["choices"][0]["message"]["content"]
    print(f"[{time.strftime('%H:%M:%S')}] Response ({elapsed:.1f}s):", file=sys.stderr, flush=True)
    print(content)
    if "usage" in resp:
        usage = resp["usage"]
        print(f"[USAGE] prompt={usage.get('prompt_tokens','?')} completion={usage.get('completion_tokens','?')} total={usage.get('total_tokens','?')}", file=sys.stderr, flush=True)
except Exception as e:
    elapsed = time.time() - start
    print(f"[{time.strftime('%H:%M:%S')}] Error after {elapsed:.1f}s: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
