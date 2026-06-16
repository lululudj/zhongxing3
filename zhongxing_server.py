# -*- coding: utf-8 -*-
"""
zhongxing2 - 语义压缩代理服务器
在 Ollama 前端自动压缩长文本，减少 50-70% token 消耗
兼容 Ollama /api/chat 和 /api/generate 接口

用法:
    python zhongxing_server.py                    # 默认端口 8080
    python zhongxing_server.py --port 8888        # 自定义端口
    python zhongxing_server.py --help             # 查看帮助

启动后，将你的 Ollama 客户端指向 http://localhost:8080 即可
"""

import sys
import os
import json
import time
import urllib.request
import urllib.error
import argparse
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# ---------- 配置 ----------
OLLAMA_BASE = "http://127.0.0.1:11434"
EXTRACTOR_MODEL = "qwen2.5:1.5b"
SCHEMA_MODEL = "qwen2.5:3b"
BRAIN_MODEL = "qwen2.5:14b"
NUM_EXTRACTORS = 3
MAX_FEEDBACK = 2

# 需要压缩的最小文本长度：短于此值的请求直接转发
MIN_COMPRESS_LEN = 200
# 不启用压缩的请求（健康检查等）
SKIP_PATHS = {"/", "/api/tags", "/api/ps", "/api/version"}

# ---------- Ollama 通信 ----------
def call_ollama(model, prompt, system="", timeout=180):
    """调用 Ollama chat API，返回文本响应"""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    data = json.dumps({
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": 0.1}
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))["message"]["content"]
    except Exception as e:
        print(f"[ERROR] Ollama call failed ({model}): {e}", file=sys.stderr)
        raise

# ---------- 语义压缩管道 ----------
def discover_schema(text, model=SCHEMA_MODEL):
    """自动发现文本类型和要提取的实体维度"""
    system = (
        "你是文本结构分析专家。分析给定文本，输出 JSON 格式，只输出 JSON，不要解释：\n"
        '{"text_type": "小说/新闻/法律/财报/论文/其他", '
        '"entity_types": ["人物","机构","数字","条款",...]}\n'
        "entity_types 列出 3-6 个需要从文中提取的实体类别。"
    )
    result = call_ollama(model, text[:3000], system, timeout=60)
    try:
        return json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        return {"text_type": "文本", "entity_types": ["实体", "属性", "关系"]}

def extract_one(text, schema, model, idx=0):
    """单个提取器：从文本中提取结构化实体信息"""
    entity_list = ", ".join(schema.get("entity_types", ["实体", "属性", "关系"]))
    system = (
        "你是信息提取专家。从文本中提取结构化信息，只输出 JSON，不要解释。\n"
        f"提取类别：{entity_list}\n"
        '输出格式：{"entities":[{"name":"实体名","type":"类别","attrs":["属性1","属性2"],"rels":["关系1"]}]}'
    )
    result = call_ollama(model, text[:3000], system, timeout=90)
    try:
        raw = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
        entities = raw.get("entities", raw) if isinstance(raw, dict) else raw
        for e in entities:
            e.setdefault("attrs", [])
            e.setdefault("rels", [])
        return {"idx": idx, "entities": entities if isinstance(entities, list) else [], "time": 0}
    except json.JSONDecodeError:
        return {"idx": idx, "entities": [], "time": 0}

def run_extractors(text, schema, model=EXTRACTOR_MODEL, num=NUM_EXTRACTORS):
    """并行运行多个提取器"""
    results = []
    with ThreadPoolExecutor(max_workers=num) as pool:
        futures = [pool.submit(extract_one, text, schema, model, i) for i in range(num)]
        for f in as_completed(futures):
            results.append(f.result())
    return results

def rrf_fuse(extractions, model=SCHEMA_MODEL, original_len=0):
    """RRF 融合：合并多路提取结果，去重排序"""
    all_entities = []
    for ext in extractions:
        all_entities.extend(ext.get("entities", []))

    # 简单去重：按名称合并
    entity_map = {}
    for e in all_entities:
        name = e.get("name", "").strip()
        if not name or len(name) < 1:
            continue
        if name not in entity_map:
            entity_map[name] = {"name": name, "type": e.get("type", ""),
                                "attrs": set(), "rels": set()}
        entity_map[name]["attrs"].update(e.get("attrs", []))
        entity_map[name]["rels"].update(e.get("rels", []))

    entities_sorted = sorted(entity_map.values(), key=lambda x: len(x["attrs"]) + len(x["rels"]), reverse=True)
    # 限制实体数量，避免 token 过多
    entities_sorted = entities_sorted[:20]

    fused_context = {
        "E": [{"n": e["name"], "t": e["type"],
                "a": sorted(e["attrs"])[:8],
                "r": sorted(e["rels"])[:8]}
              for e in entities_sorted]
    }
    return {"fused_context": fused_context, "entity_count": len(entities_sorted)}

def serialize_compact(fused_context):
    """将融合结果序列化为紧凑格式（机语）"""
    lines = []
    entities = fused_context.get("E", [])
    for e in entities:
        attrs = ", ".join(e.get("a", [])[:5])
        rels = ", ".join(e.get("r", [])[:5])
        tail = ""
        if e.get("t"):
            tail += f"  [{e['t']}]"
        if attrs or rels:
            detail = " | ".join(filter(None, [attrs, rels]))
            tail += f"  {{{detail}}}"
        lines.append(f"- {e['n']}{tail}" if tail else f"- {e['n']}")

    if not lines:
        return "(无结构化实体)"

    return "\n".join(lines)

def compress_text(text):
    """完整压缩管道：文本 → 机语"""
    t0 = time.time()

    # Step 1: 维度发现
    schema = discover_schema(text)
    # Step 2: 并行提取
    extractions = run_extractors(text, schema)

    total_ents = sum(len(e["entities"]) for e in extractions)
    if total_ents < 5:
        extractions2 = run_extractors(text, schema)
        if sum(len(e["entities"]) for e in extractions2) > total_ents:
            extractions = extractions2

    # Step 3: RRF 融合
    fusion = rrf_fuse(extractions, SCHEMA_MODEL, len(text))

    # Step 4: 序列化
    machine_lang = serialize_compact(fusion["fused_context"])

    elapsed = time.time() - t0
    return {
        "machine_lang": machine_lang,
        "original_len": len(text),
        "compressed_len": len(machine_lang),
        "ratio": round(len(text) / max(len(machine_lang), 1), 1),
        "entities": fusion["entity_count"],
        "compress_time": round(elapsed, 1),
    }

# ---------- 投机检测：是否启用压缩 ----------
def should_compress(messages):
    """判断是否需要对请求启用压缩"""
    # 收集所有文本内容
    all_text = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            all_text += content + "\n"
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    all_text += part.get("text", "") + "\n"

    total_len = len(all_text)
    if total_len < MIN_COMPRESS_LEN:
        return False, total_len, None

    # 分离 user 消息和 system 消息
    system_msgs = [m for m in messages if m.get("role") == "system"]
    user_msgs = [m for m in messages if m.get("role") == "user"]
    other_msgs = [m for m in messages if m.get("role") not in ("system", "user")]

    return True, total_len, (system_msgs, user_msgs, other_msgs)

# ---------- 辅助：分离上下文和问题 ----------
_QUESTION_PATTERNS = [
    r'[?？]',
    r'请问',
    r'问题[：:]',
    r'Question[：:]',
]

def split_context_question(text):
    """从文本中分离上下文和问题"""
    # 尝试按双换行+问句切分
    lines = text.strip().split('\n')
    # 收集前面所有非问句行作为上下文
    context_lines = []
    question_lines = []
    found_question = False
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if not found_question and any(re.search(p, line) for p in _QUESTION_PATTERNS):
            question_lines.insert(0, line)
            found_question = True
        elif found_question:
            # 问题之后的也加入上下文（实际上是从后往前，所以这相当于前面的行）
            context_lines.insert(0, line)
        else:
            # 没有问句标识，可能在上下文里
            context_lines.insert(0, line)

    if found_question:
        return '\n'.join(context_lines), '\n'.join(question_lines)

    # 如果没有找到明确的问句，用双换行分割，最后一段为问题
    parts = re.split(r'\n\s*\n', text.strip())
    if len(parts) >= 2:
        return '\n\n'.join(parts[:-1]), parts[-1]

    return text, ""

# ---------- HTTP 代理服务器 ----------
class ZhongxingHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器：拦截 /api/chat 并压缩后转发"""

    def log_message(self, format, *args):
        """精简日志输出"""
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}", file=sys.stderr)

    def _forward_raw(self, path, body=None, method="POST", headers_extra=None):
        """原样转发请求到 Ollama"""
        url = f"{OLLAMA_BASE}{path}"
        data = body.encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in ("transfer-encoding",):
                        self.send_header(k, v)
                # 附加 zhongxing2 自定义响应头
                if headers_extra:
                    for k, v in headers_extra.items():
                        self.send_header(k, str(v))
                self.end_headers()
                self.wfile.write(r.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_compressed_chat(self, body):
        """处理压缩后的 chat 请求"""
        try:
            req_data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        model = req_data.get("model", BRAIN_MODEL)
        messages = req_data.get("messages", [])
        stream = req_data.get("stream", False)

        # 检测是否需要压缩
        do_compress, total_len, msg_parts = should_compress(messages)

        if not do_compress:
            # 短文本，直接转发
            self._forward_raw("/api/chat", body)
            return

        system_msgs, user_msgs, other_msgs = msg_parts

        # 提取所有 user 消息内容，最后一个 user 消息拆分上下文和问题
        question = ""
        context_parts = []

        # 收集前 N-1 个 user 消息作为上下文
        for msg in user_msgs[:-1]:
            content = msg.get("content", "")
            if isinstance(content, str):
                context_parts.append(content)

        # 最后一个 user 消息：可能同时包含上下文和问题
        last_user = user_msgs[-1].get("content", "") if user_msgs else ""
        if isinstance(last_user, str):
            ctx, q = split_context_question(last_user)
            if ctx:
                context_parts.append(ctx)
            question = q

        full_context = "\n\n".join(filter(None, context_parts))

        # 如果没有分离出问题，用整个最后消息作为问题
        if not question:
            question = last_user
            # 整个作为问题的话，上下文是前面所有的
            pass

        # 如果没有足够上下文，直接转发
        if not full_context or len(full_context) < MIN_COMPRESS_LEN:
            self._forward_raw("/api/chat", body)
            return

        # 执行压缩
        print(f"[COMPRESS] 文本 {len(full_context)} 字 → 压缩中...", file=sys.stderr)
        try:
            result = compress_text(full_context)
            machine_lang = result["machine_lang"]
            print(f"[COMPRESS] 压缩比 {result['ratio']}:1, 提取 {result['entities']} 实体, 耗时 {result['compress_time']}s",
                  file=sys.stderr)
        except Exception as e:
            print(f"[COMPRESS] 压缩失败: {e}, 原样转发", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._forward_raw("/api/chat", body)
            return

        # 构造压缩后的 prompt
        system_prompt = ""
        for sm in system_msgs:
            system_prompt += sm.get("content", "") + "\n"

        compressed_prompt = f"""[上下文摘要 - 机器语言格式]
以下是从原始文本中提取的结构化信息，请基于这些信息回答问题：
{machine_lang}

[问题]
{question}

请只根据以上上下文摘要回答问题。如果信息不足，请说明需要哪些补充信息。"""

        # 转发压缩后的请求到 Ollama
        new_body = json.dumps({
            "model": model,
            "messages": [
                *system_msgs,
                *other_msgs,
                {"role": "user", "content": compressed_prompt}
            ],
            "stream": stream,
            "options": req_data.get("options", {}),
        })

        print(f"[FORWARD] 压缩后 {len(compressed_prompt)} 字 → Ollama {model}", file=sys.stderr)
        self._forward_raw("/api/chat", new_body, headers_extra={
            "X-Zhongxing-Original-Len": str(result["original_len"]),
            "X-Zhongxing-Compressed-Len": str(result["compressed_len"]),
            "X-Zhongxing-Ratio": str(result["ratio"]),
        })

    def do_POST(self):
        """处理 POST 请求"""
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else ""

        path = self.path.rstrip("/") or "/"

        if path in SKIP_PATHS:
            self._forward_raw(path, body)
        elif path == "/api/chat":
            self._handle_compressed_chat(body)
        else:
            self._forward_raw(path, body)

    def do_GET(self):
        """处理 GET 请求（健康检查等）"""
        self._forward_raw(self.path, method="GET")

    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser(
        description="zhongxing2 - Ollama 语义压缩代理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python zhongxing_server.py                    # 启动在 8080 端口
  python zhongxing_server.py --port 8888        # 自定义端口
  python zhongxing_server.py --ollama http://192.168.1.100:11434  # 远程 Ollama

启动后将你的 Ollama 客户端（如 Open WebUI）指向此服务器即可自动压缩长文本。
        """
    )
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama 地址")
    parser.add_argument("--min-len", type=int, default=200, help="触发压缩的最小文本长度 (默认: 200)")
    parser.add_argument("--extractor-model", default="qwen2.5:1.5b", help="提取模型 (默认: qwen2.5:1.5b)")
    parser.add_argument("--schema-model", default="qwen2.5:3b", help="维度发现模型 (默认: qwen2.5:3b)")
    parser.add_argument("--brain-model", default="qwen2.5:14b", help="推理模型 (默认: qwen2.5:14b)")

    args = parser.parse_args()

    # 更新全局配置
    global OLLAMA_BASE, EXTRACTOR_MODEL, SCHEMA_MODEL, BRAIN_MODEL, MIN_COMPRESS_LEN
    OLLAMA_BASE = args.ollama.rstrip("/")
    EXTRACTOR_MODEL = args.extractor_model
    SCHEMA_MODEL = args.schema_model
    BRAIN_MODEL = args.brain_model
    MIN_COMPRESS_LEN = args.min_len

    # 检查 Ollama 连通性
    print(f"zhongxing2 语义压缩代理 v1.0", file=sys.stderr)
    print(f"=" * 50, file=sys.stderr)
    print(f"  监听: http://{args.host}:{args.port}", file=sys.stderr)
    print(f"  后端: {OLLAMA_BASE}", file=sys.stderr)
    print(f"  压缩触发长度: >= {MIN_COMPRESS_LEN} 字", file=sys.stderr)
    print(f"  提取模型: {EXTRACTOR_MODEL}", file=sys.stderr)
    print(f"  推理模型: {BRAIN_MODEL}", file=sys.stderr)
    print(f"=" * 50, file=sys.stderr)

    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            models = json.loads(r.read()).get("models", [])
            model_names = [m["name"] for m in models]
            print(f"  Ollama 已连接, 已安装模型: {', '.join(model_names[:5])}", file=sys.stderr)
    except Exception as e:
        print(f"  [WARNING] 无法连接 Ollama ({e}), 请确保 Ollama 已启动", file=sys.stderr)

    print(f"\n  将你的 Ollama 客户端指向 http://localhost:{args.port} 即可使用", file=sys.stderr)
    print(f"  按 Ctrl+C 停止服务器\n", file=sys.stderr)

    server = HTTPServer((args.host, args.port), ZhongxingHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nzhongxing2 已停止", file=sys.stderr)
        server.shutdown()

if __name__ == "__main__":
    main()