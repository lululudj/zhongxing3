# -*- coding: utf-8 -*-
"""
zhongxing 1.7.0 — 通用压缩管道 (常驻加速+懒鹰眼+精简prompt)
v1.6.3→v1.7.0升级:
1. 模型常驻: 1.5B/3B keep_alive=3600s，避免反复加载卸载 → 省2s/题
2. 懒鹰眼: 高置信度短答案跳过校验 → 省1.5-2s/题(40%触发)
3. 精简Prompt: 鹰眼/大脑/提取prompt缩减30-50% token → 省0.5s/调用
4. 3B不卸载: 鹰眼后保持3B常驻，不执行unload+wait
5. 预期提速: 8.7s→5-6s/题 (35-40%加速)

v1.6.2→v1.6.3升级:
1. 鹰眼校验Agent: 3B模型纵观全局，检测答案矛盾/歧义/遗漏并修正
   → eagle_eye_validate() / big_brain_answer集成eagle_eye参数
2. 校验四维度: 事实一致性 / 逻辑自洽性 / 信息充分性 / 歧义检测
3. 克莱因瓶理念: 鹰眼构成答案→校验→修正的闭环反馈流
4. --no-eagle-eye 参数可禁用鹰眼校验
5. 交互模式支持鹰眼校验状态显示

v1.6.1→v1.6.2升级:
1. 莫比乌斯环VRAM折叠: 大模型(14B)推理前自动卸载小模型，推理后释放
2. 动态GPU层分配: 根据剩余VRAM计算最优num_gpu，适配GTX 1060 6GB
3. --no-vram-folding / --vram-limit 参数
"""

import urllib.request
import urllib.error
import json
import time
import sys
import argparse
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# ============ 配置 ============
OLLAMA_BASE = "http://127.0.0.1:11434"

# V4-Flash配置 (antirez llama.cpp fork, 子进程调用)
V4FLASH_CLI = r"E:\zhongxing2\llama-cpp-v4flash\build\bin\Release\llama-cli.exe"
V4FLASH_MODEL = r"E:\models\v4-flash-download\deepseek-v4-flash-iq2xxs.gguf"
V4FLASH_SERVER = "http://127.0.0.1:8080"

# 垃圾实体名黑名单(1.5b有时把prompt模板"n":"名"字面输出)
SPURIOUS_NAMES = {"名", "的", "了", "是", "在", "和", "与", "有", "为",
                  "其", "他", "她", "它", "这", "那", "不", "也", "都",
                  "被", "把", "让", "给", "向", "从", "到", "又", "再",
                  "则", "而", "或", "但", "却", "已", "曾", "将", "会"}

FALLBACK_CONFIG = {
    "extractor_model": "qwen2.5:1.5b",       # 通用提取器(3个并行)
    "schema_model": "qwen2.5:3b",             # 维度自发现+NER (3b列人名更准)
    "big_brain_model": "qwen2.5:14b",         # 推理
    "eagle_eye_model": "qwen2.5:3b",          # 鹰眼校验(与schema共用3B)
    "num_extractors": 3,
    "max_feedback_rounds": 2,
    "vram_folding": True,                     # 启用VRAM折叠（莫比乌斯环模式）
    "vram_limit_mb": 6144,                    # 目标VRAM上限(6GB for GTX1060)
    "per_layer_vram": 200,                    # 每层预估VRAM(MiB)
    "eagle_eye_enabled": True,                # 启用鹰眼校验
}

# ============ VRAM折叠模块（莫比乌斯环模式） ============
def nvidia_smi():
    """获取GPU显存状态 (used, free, total) MiB"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
             '--format=csv,noheader,nounits'], timeout=10).decode().strip()
        used, free, total = [int(x.strip()) for x in out.split(',')]
        return used, free, total
    except Exception:
        return None, None, None

def unload_model(model_name: str):
    """卸载指定模型，释放VRAM"""
    try:
        body = json.dumps({"model": model_name, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/generate",
            data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False

def get_optimal_gpu_layers(vram_limit: int, per_layer: int = 200) -> int:
    """根据当前剩余VRAM计算最优GPU层数"""
    used, free, total = nvidia_smi()
    if free is None:
        return 100  # 无法检测，全部加载
    
    # 保留10%安全余量
    safe_free = int(free * 0.9)
    max_layers_by_vram = max(1, safe_free // per_layer)
    
    # 不低于4层(保证基本GPU加速)，不高于vram_limit能容纳的
    min_layers = 4
    return max(min_layers, max_layers_by_vram)

def vram_aware_call(model: str, prompt: str, system: str = "", timeout: int = 180,
                    vram_folding: bool = True, vram_limit: int = 6144,
                    keep_alive: int = None) -> str:
    """VRAM折叠包装器: 自动管理模型生命周期"""
    # V4-Flash路由: 直接走llama-cli子进程(太慢, 暂用子进程代替server)
    if "v4flash" in model.lower() or "v4-flash" in model.lower() or "deepseek-v4" in model.lower():
        print(f"  [V4-Flash] 子进程推理中(284B CPU)...")
        return call_v4flash_cli(prompt, system, max_tokens=500, timeout=1800)

    if not vram_folding:
        return call_ollama(model, prompt, system, timeout, keep_alive=keep_alive)
    
    used, free, total = nvidia_smi()
    
    is_big = any(x in model for x in ["14b", "7b", "13b", "v4", "flash", "deepseek"])
    if is_big:
        # 莫比乌斯环: 小模型常驻不卸载，只给14B腾GPU层数
        is_small_loaded = True  # 假设1.5B+3B已常驻，不卸载
        optimal_layers = get_optimal_gpu_layers(vram_limit)
        options = {"num_gpu": optimal_layers}
        if vram_folding:
            print(f"  [VRAM折叠] 14B加载{optimal_layers}层GPU(剩余{free}MiB)，小模型常驻")
    else:
        options = None
    
    return call_ollama(model, prompt, system, timeout, options, keep_alive=keep_alive)

def fold_cleanup(models: list = None):
    """清理所有模型，释放VRAM"""
    if models is None:
        models = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:14b"]
    for m in models:
        unload_model(m)
    time.sleep(2)

# ============ V4-Flash CLI子进程调用 ============
def call_v4flash_cli(prompt: str, system: str = "", max_tokens: int = 500,
                     timeout: int = 1800) -> str:
    """通过llama-cli子进程调用V4-Flash。模型加载慢(~2min)，推理更慢(CPU 284B)"""
    import os, tempfile

    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    cmd = [
        V4FLASH_CLI,
        "-m", V4FLASH_MODEL,
        "-p", full_prompt,
        "-n", str(max_tokens),
        "-c", "1024",
        "--no-display-prompt",
        "--temp", "0.7",
        "-t", "4",
        "--log-disable",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(V4FLASH_CLI),
        )
        stdout = proc.stdout

        # 提取生成文本：在[Start thinking]之后，下一段非空文本
        lines = stdout.split('\n')
        in_response = False
        result_lines = []
        for line in lines:
            stripped = line.strip()
            if 'Start thinking' in stripped:
                in_response = True
                continue
            if in_response:
                if 'available commands:' in stripped:
                    break
                if stripped:
                    result_lines.append(line)

        return '\n'.join(result_lines).strip() or "(空输出)"
    except subprocess.TimeoutExpired:
        return f"ERROR: V4-Flash推理超时({timeout}s)"
    except Exception as e:
        return f"ERROR: V4-Flash调用失败 - {e}"

# ============ Ollama调用 ============
def call_ollama(model: str, prompt: str, system: str = "", timeout: int = 180,
                options: dict = None, keep_alive: int = None) -> str:
    """keep_alive: 秒数，模型用完后保持加载状态。None=使用Ollama默认(5分钟)"""
    try:
        is_reasoning_model = "deepseek-r1" in model
        is_minicpm = "minicpm" in model.lower()
        msgs = []
        if system and not is_reasoning_model:
            msgs.append({"role": "system", "content": system})
        if is_minicpm:
            user_content = f"{prompt}\n/no_think"
        elif is_reasoning_model:
            user_content = f"{system}\n{prompt}" if system else prompt
        else:
            user_content = prompt
        msgs.append({"role": "user", "content": user_content})

        body = {"model": model, "messages": msgs, "stream": False}
        merged_options = dict(options) if options else {}
        if keep_alive is not None:
            merged_options["num_keep"] = 0  # 不要num_keep影响
        if merged_options:
            body["options"] = merged_options
        if keep_alive is not None:
            body["keep_alive"] = keep_alive  # 秒数，Ollama原生支持

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.loads(r.read().decode("utf-8"))["message"]["content"]
        if "</think" in content:
            content = content.split("</think")[-1].strip()
        return content
    except urllib.error.URLError as e:
        print(f"[ERROR] Ollama连接失败 ({OLLAMA_BASE}): {e}")
        return f"ERROR: 连接失败 - {e}"
    except Exception as e:
        print(f"[ERROR] 调用 {model} 失败: {e}")
        return f"ERROR: {e}"


def try_parse_json(raw: str):
    for extract_fn in [
        lambda: json.loads(raw),
        lambda: json.loads(raw[raw.index("["):raw.rindex("]")+1]) if "[" in raw and "]" in raw else None,
        lambda: json.loads(raw[raw.index("{"):raw.rindex("}")+1]) if "{" in raw and "}" in raw else None,
    ]:
        try:
            result = extract_fn()
            if result is not None:
                return result, True
        except:
            pass
    return None, False


# ============ 通用 人语→机语 压缩 ============

def normalize_items(items) -> list:
    """归一化A/R/C字段: str→单元素list, dict→提取文本, 单字符过滤"""
    if items is None:
        return []
    if isinstance(items, str):
        # 1.5b有时输出"C": "灵魂遁入骨灵冷火"(字符串而非列表)
        # Python遍历字符串会逐字拆开→必须归一化
        return [items] if len(items) >= 2 else []
    if isinstance(items, dict):
        # 1.5b有时输出嵌套dict在R字段里
        return [str(v) for v in items.values() if v and len(str(v)) >= 2]
    if isinstance(items, list):
        result = []
        for item in items:
            if isinstance(item, str):
                if len(item) >= 2:
                    result.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    vs = str(v).strip()
                    if len(vs) >= 2:
                        result.append(vs)
            elif item is not None:
                s = str(item).strip()
                if len(s) >= 2:
                    result.append(s)
        return result
    return []


def substring_dedup(items: list) -> list:
    """子串去重: 如果短项是长项的子串，删短留长(信息量更大)
    例: ["叛徒","叛徒弟子"] → ["叛徒弟子"]
    """
    if not items:
        return items
    str_items = [(str(i).strip(), i) for i in items if str(i).strip()]
    result = []
    for s, orig in str_items:
        # 检查s是否已被某个更长项包含
        dominated = False
        for rs, _ in result:
            if s in rs and len(s) < len(rs):
                dominated = True
                break
        if not dominated:
            # 移除已被s包含的更短项
            result = [(rs, ro) for rs, ro in result if not (rs in s and len(rs) < len(s))]
            result.append((s, orig))
    return [orig for _, orig in result]


def compactify(s: str) -> str:
    """通用 人语→机语 确定性转换
    核心原则: 删除不改变语义的冗余表述，保留所有事实信息
    适用于小说/论文/法律/财报等各类型文本
    """
    if not s:
        return s

    # === 1. 去除虚词/判断词 ===
    for word in ['乃是', '则是', '即为', '亦为', '便是', '所谓',
                 '拥有', '正是', '不愧是', '算得上']:
        s = s.replace(word, '')

    # === 2. 去除程度副词 (不改变事实) ===
    for word in ['极其', '非常', '特别', '相当', '十分', '尤为',
                 '最为', '格外', '异常', '无比', '极', '甚', '颇',
                 '了极大', '极大', '巨大']:
        s = s.replace(word, '')

    # === 3. 去除"X的"结构 (X为通用修饰词) ===
    for pattern in ['强大的', '特殊的', '过人的', '惊人的', '巨大的',
                    '重要的', '核心的', '关键的', '主要的', '著名的',
                    '最大的', '最强的', '最高的', '极强的', '深厚的',
                    '强大的', '极其强大的']:
        s = s.replace(pattern, '')

    # === 4. 压缩等级/身份模式 ===
    s = re.sub(r'(\w{1,4})级别的强者', r'\1', s)
    s = re.sub(r'(\w{1,4})级别的', r'\1', s)
    s = re.sub(r'(\w{1,4})等级的', r'\1', s)

    # === 5. 因果链压缩 ===
    s = re.sub(r'导致(其|他|她|它)?', '→', s)
    s = re.sub(r'使得(他|她|它)?', '→', s)
    s = re.sub(r'引起(了)?', '→', s)
    s = re.sub(r'造成(了)?', '→', s)
    s = re.sub(r'从而', '→', s)
    s = re.sub(r'以至于', '→', s)

    # === 6. 被动/迫使压缩 ===
    s = s.replace('被迫', '')
    s = s.replace('遭到', '被')
    s = s.replace('受到', '被')

    # === 7. 动词短语压缩 ===
    s = s.replace('不修炼也能', '免修')
    s = s.replace('自动吸收', '吸')
    s = s.replace('提升实力', '↑实力')
    s = s.replace('失去控制', '失控')
    s = s.replace('反噬失控', '反噬')
    s = s.replace('掌控着', '控')
    s = s.replace('掌控', '控')
    s = s.replace('能够', '可')
    s = s.replace('可以', '可')
    s = s.replace('不可', '不可')  # 防止被上面覆盖
    s = s.replace('但是同时', '但')
    s = s.replace('但是', '但')
    s = s.replace('然而', '但')
    s = s.replace('不过', '但')
    s = s.replace('同时也意味着', '且')
    s = s.replace('同时也', '且')
    s = s.replace('也意味着', '即')
    s = s.replace('意味着', '即')
    s = s.replace('这种特殊体质', '')
    s = s.replace('这种体质', '')
    s = s.replace('这种', '')
    s = s.replace('那样', '')

    # === 8. 尾缀清理 ===
    s = re.sub(r'(之中|之内|之上|之下|之际|之后|之前|之间)(?=([，,；;。]|$))', '', s)
    s = s.rstrip('。；：，、')

    # === 9. 多余标点/空格清理 ===
    s = re.sub(r'[，,]{2,}', ',', s)
    s = re.sub(r'→{2,}', '→', s)
    s = re.sub(r'\s{2,}', ' ', s)
    # 清理"的"在句尾
    s = re.sub(r'的(?=[，,；;|]|$)', '', s)

    return s.strip()


def compactify_entity(ent: dict) -> dict:
    """对单个实体的所有属性字段做紧凑化"""
    result = {"n": ent.get("n", "")}
    for key in ["A", "R", "C"]:
        items = normalize_items(ent.get(key, []))
        if not items:
            continue
        compacted = []
        for item in items:
            s = compactify(str(item))
            if s and len(s) >= 2:
                compacted.append(s)
        compacted = substring_dedup(compacted)
        if compacted:
            result[key] = compacted
    return result


# ============ 紧凑序列化 ============
def serialize_one_compact(ent: dict) -> str:
    """序列化单个实体为紧凑行格式"""
    name = ent.get("n", "")
    if not name:
        return ""
    a = ",".join(normalize_items(ent.get("A", [])))
    r = ",".join(normalize_items(ent.get("R", [])))
    c = ",".join(normalize_items(ent.get("C", [])))
    return f"[{name}] {a} | {r} | {c}"


def serialize_compact(fused_context: dict) -> str:
    """紧凑行格式: [实体名] 属性 | 关系 | 因果
    比JSON省50%+字符(去掉key名/引号/括号开销)
    14b模型可零损耗读取
    """
    entities = fused_context.get("E", [])
    if not entities:
        return ""
    lines = []
    for ent in entities:
        line = serialize_one_compact(ent)
        if line:
            lines.append(line)
    return "\n".join(lines)


# ============ Layer 0: 维度自发现 (可选) ============
def discover_schema(text: str, model: str) -> Dict:
    """让小模型自动推断文本的潜在维度/结构"""
    prompt = (
        f"分析文本结构，回答:\n"
        f"1. 文本类型(小说/论文/法律/财报/新闻/其他)\n"
        f"2. 核心实体类型(如:人物/机构/法条/指标)，逗号分隔\n"
        f"3. 核心属性类型(如:实力/关系/职责/数值)，逗号分隔\n"
        f"4. 关键关系类型(如:师徒/隶属/引用/对比)，逗号分隔\n"
        f"格式: 类型|实体类型|属性类型|关系类型\n"
        f"文本: {text[:3000]}"
    )
    raw = call_ollama(model, prompt, "文本结构分析。只输出一行4段信息。")

    parts = raw.strip().split("|")
    if len(parts) < 4:
        return {
            "text_type": "通用",
            "entity_types": ["实体"],
            "attr_types": ["属性", "关系", "因果"],
            "rel_types": ["关系"]
        }

    return {
        "text_type": parts[0].strip(),
        "entity_types": [x.strip() for x in parts[1].split(",") if x.strip()],
        "attr_types": [x.strip() for x in parts[2].split(",") if x.strip()],
        "rel_types": [x.strip() for x in parts[3].split(",") if x.strip()],
    }


# ============ Layer 1: 通用提取器(3个并行) ============
def universal_extract(text: str, extractor_id: int, schema: Dict, model: str,
                      question: str = "") -> Dict:
    """通用提取: 根据schema和问题动态调整提取策略
    v2.0优化: 去掉'只提人物'限制，支持科技/法律/医学等多领域实体
    """
    entity_hint = "/".join(schema.get("entity_types", ["实体"]))
    attr_hint = "/".join(schema.get("attr_types", ["属性", "数值", "时间"]))
    rel_hint = "/".join(schema.get("rel_types", ["关系", "因果"]))
    
    # 根据entity_types判断是否为小说类
    entity_types = schema.get("entity_types", [])
    is_novel = "人物" in entity_types and len(entity_types) <= 3
    
    # 问题导向提示（如果有问题）
    question_hint = f"\n关注问题关键词: {question[:100]}" if question else ""
    
    if is_novel:
        # 小说类：保持原有规则
        prompt = (
            f"提取文本关键信息→JSON。\n"
            f"实体类型: {entity_hint} | 属性: {attr_hint} | 关系: {rel_hint}\n"
            f"规则:\n"
            f"1. 每个实体一条，n写真名(韩枫不是药老的弟子)\n"
            f"2. A写短词(斗皇/厄难毒体/家主)，不写长句\n"
            f"3. R写短词(师徒/朋友/叛徒弟子)\n"
            f"4. C用→连(偷袭→灵魂遁入骨灵冷火)\n"
            f"5. 只提人物，不提地点/组织/物品\n"
            f"6. 逐句扫描，不漏人物{question_hint}\n"
            f'输出: {{"E":[{{"n":"名","R":[...],"A":[...],"C":[...]}}]}}\n'
            f"文本: {text[:4000]}"
        )
        system = f"提取器{extractor_id}。逐句扫描不漏人。n写真名。A写短词。只输出JSON。"
    else:
        # 通用/科技/法律/医学类：放宽实体限制
        prompt = (
            f"提取文本关键信息→JSON。\n"
            f"实体类型: {entity_hint}\n"
            f"属性类型: {attr_hint}\n"
            f"关系类型: {rel_hint}\n"
            f"规则:\n"
            f"1. n写真名(人物/机构/概念/产品/技术都可)\n"
            f"2. A写属性值(架构名/数值/时间/参数)，用短词\n"
            f"3. R写关系(客户/属于/包含/开发)\n"
            f"4. C写因果/过程(发布→拥有→占比)\n"
            f"5. 数字/参数/年份都要提取{question_hint}\n"
            f'输出: {{"E":[{{"n":"名","A":[...],"R":[...],"C":[...]}}]}}\n'
            f"文本: {text[:4000]}"
        )
        system = f"提取器{extractor_id}。提取所有实体(人物/机构/概念/产品)。A写短词。只输出JSON。"

    start = time.time()
    raw = call_ollama(model, prompt, system, keep_alive=3600)  # 1.5B常驻1小时
    elapsed = time.time() - start

    parsed, ok = try_parse_json(raw)
    if not ok:
        return {"extractor_id": extractor_id, "entities": [], "raw": raw[:200], "time": round(elapsed, 1)}

    if isinstance(parsed, dict):
        entities = parsed.get("E", [])
    elif isinstance(parsed, list):
        entities = parsed
    else:
        entities = []

    return {"extractor_id": extractor_id, "entities": entities, "raw": raw[:200], "time": round(elapsed, 1)}


def chunk_text(text: str, chunk_size: int = 900) -> List[str]:
    """将长文本按段落边界分块，每块~chunk_size字"""
    if len(text) <= chunk_size:
        return [text]
    
    # 先按双换行分段
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = p
        else:
            current = current + "\n\n" + p if current else p
    if current.strip():
        chunks.append(current.strip())
    
    # 如果某块仍然过长，按单换行再切
    final_chunks = []
    for c in chunks:
        if len(c) <= chunk_size:
            final_chunks.append(c)
        else:
            lines = c.split('\n')
            sub = ""
            for line in lines:
                if len(sub) + len(line) + 1 > chunk_size and sub:
                    final_chunks.append(sub.strip())
                    sub = line
                else:
                    sub = sub + "\n" + line if sub else line
            if sub.strip():
                final_chunks.append(sub.strip())
    
    return final_chunks if final_chunks else [text]


def run_extractors(text: str, schema: Dict, model: str, num: int = 3) -> List[Dict]:
    """并行运行N个通用提取器，0实体自动重试1次
    长文本(>1200字)自动分块，每块独立提取后合并
    """
    chunks = chunk_text(text)
    all_results = []
    
    if len(chunks) > 1:
        print(f"  长文本分块: {len(text)}字 → {len(chunks)}块")
    
    for ci, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  --- 块{ci+1}/{len(chunks)} ({len(chunk)}字) ---")
        
        results = []
        with ThreadPoolExecutor(max_workers=num) as executor:
            futures = {
                executor.submit(universal_extract, chunk, i+1, schema, model): i+1
                for i in range(num)
            }
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    result = future.result()
                    ent_count = len(result["entities"])
                    print(f"  [提取器{eid}] {result['time']}s, {ent_count}实体")
                    # 0实体自动重试1次
                    if ent_count == 0:
                        retry = universal_extract(chunk, eid, schema, model)
                        retry_count = len(retry["entities"])
                        print(f"  [提取器{eid}] 重试: {retry['time']}s, {retry_count}实体")
                        if retry_count > 0:
                            result = retry
                    results.append(result)
                except Exception as e:
                    print(f"  [提取器{eid}] 失败: {e}")
                    results.append({"extractor_id": eid, "entities": [], "raw": str(e), "time": 0})
        
        # 给不同块的提取器加偏移ID，避免RRF合并时extractor_id冲突
        for r in results:
            r["extractor_id"] = r["extractor_id"] + ci * 100
        all_results.extend(results)
    
    # Pipeline级重试: 总实体<5时重跑一次
    total_ents = sum(len(e["entities"]) for e in all_results)
    if total_ents < 5:
        print(f"  总实体{total_ents}<5，重跑提取...")
        retry_results = []
        for ci, chunk in enumerate(chunks):
            with ThreadPoolExecutor(max_workers=num) as executor:
                futures = {
                    executor.submit(universal_extract, chunk, i+1, schema, model): i+1
                    for i in range(num)
                }
                for future in as_completed(futures):
                    eid = futures[future]
                    try:
                        result = future.result()
                        result["extractor_id"] = result["extractor_id"] + ci * 100 + 50
                        retry_results.append(result)
                    except:
                        pass
        total_retry = sum(len(e["entities"]) for e in retry_results)
        if total_retry > total_ents:
            all_results = retry_results
            print(f"  重跑更好: {total_retry}实体 > {total_ents}实体")
    
    all_results.sort(key=lambda r: r["extractor_id"])
    return all_results


# ============ Layer 2: RRF融合 (纯确定性) ============
def rrf_fuse(extractions: List[Dict], model: str, original_len: int) -> Dict:
    """
    RRF融合 + 确定性紧凑化 + 紧凑行格式序列化
    不调用任何模型(3b不可信: 删实体/篡改事实/展开属性)
    """
    K = 60
    rrf_start = time.time()

    # Step 1: 收集所有实体，按名字分组（模糊匹配：短名是长名的子串则合并）
    entity_scores = {}
    name_map = {}  # alias -> canonical_name
    for ext in extractions:
        for rank, ent in enumerate(ext["entities"]):
            name = ent.get("n", "").strip()
            if not name:
                continue
            # 模糊合并：如果已有实体名包含当前名，或当前名包含已有实体名，则合并
            canonical = name
            for existing_name in list(entity_scores.keys()):
                if name in existing_name or existing_name in name:
                    # 保留更短的作为标准名(韩枫 > 韩枫是药老的弟子)
                    canonical = existing_name if len(existing_name) <= len(name) else name
                    if existing_name != canonical:
                        entity_scores[canonical] = entity_scores.pop(existing_name)
                    break
            if canonical not in entity_scores:
                entity_scores[canonical] = {"rrf_score": 0, "extractors": set(), "raw_attrs": []}
            entity_scores[canonical]["rrf_score"] += 1.0 / (K + rank + 1)
            entity_scores[canonical]["extractors"].add(ext["extractor_id"])
            entity_scores[canonical]["raw_attrs"].append(ent)

    # Step 2: 按RRF分数排序
    sorted_entities = sorted(entity_scores.items(), key=lambda x: -x[1]["rrf_score"])

    # Step 3: 合并属性 + 紧凑化 + 过滤非人物实体 + 垃圾名过滤
    non_person_suffixes = ["学院", "炼气塔", "山脉", "家族", "城", "域", "塔",
                           "异火", "毒体", "心炎", "冷火", "奇物"]
    ranked_data = []
    for name, info in sorted_entities:
        # 过滤：垃圾名(1.5b把"名"字面输出) + 名字太短 + 非人物后缀
        if name in SPURIOUS_NAMES or len(name) < 2:
            continue
        if any(name.endswith(s) for s in non_person_suffixes):
            continue
        merged = {"n": name, "R": [], "A": [], "C": []}
        seen_r, seen_a, seen_c = set(), set(), set()
        for ent in info["raw_attrs"]:
            # 使用normalize_items归一化(防字符拆解)
            for r in normalize_items(ent.get("R", [])):
                if r not in seen_r:
                    merged["R"].append(r)
                    seen_r.add(r)
            for a in normalize_items(ent.get("A", [])):
                if a not in seen_a:
                    merged["A"].append(a)
                    seen_a.add(a)
            for c in normalize_items(ent.get("C", [])):
                if c not in seen_c:
                    merged["C"].append(c)
                    seen_c.add(c)
        # 去空字段
        merged = {k: v for k, v in merged.items() if v}

        # 通用紧凑化(含子串去重)
        merged = compactify_entity(merged)
        ranked_data.append(merged)

    fused = {"E": ranked_data}

    # 同时计算JSON和紧凑格式的长度，用于对比
    json_str = json.dumps(fused, ensure_ascii=False)
    compact_str = serialize_compact(fused)

    return {
        "fused_context": fused,
        "compact_context": compact_str,
        "rrf_ranking": [(n, round(info["rrf_score"], 4), len(info["extractors"])) for n, info in sorted_entities[:20]],
        "raw_output": "deterministic_compact_v2",
        "json_len": len(json_str),
        "compact_len": len(compact_str),
        "time": round(time.time() - rrf_start, 1)
    }


# ============ Layer 2b: 补漏校验 (用1.5b替代3b) ============
def entity_gap_check(text: str, fused_context, model: str) -> Dict:
    """人名完整性校验: 用1.5b列人名(简单任务)，确定性紧凑化补入E表"""
    if isinstance(fused_context, list):
        entities = fused_context
    else:
        entities = fused_context.get("E", [])

    existing_names = set()
    for ent in entities:
        name = ent.get("n", "")
        if name:
            existing_names.add(name)

    # 3b列人名(比1.5b更准，3b识别人名更稳定)
    prompt = (
        f"列出原文中所有【人物名】(只列2-4字真名，如韩枫/药老/萧炎)，不列地名/组织/物品/异火/毒体/火焰/称号，逗号分隔。\n"
        f"原文：{text[:4000]}"
    )
    raw = call_ollama(model, prompt, "只列人名(2-4字真名)，逗号分隔，不要其他内容。")

    first_line = raw.strip().split("\n")[0]
    name_candidates = [n.strip() for n in first_line.replace("，", ",").split(",") if n.strip()]
    # 过滤: 2-6字，不含地名/物名/身份词关键词
    non_person_keywords = ["学院", "炼气塔", "异火", "城", "域", "山脉", "心炎",
                           "毒体", "家族", "族", "骨灵", "冷火", "塔底", "后山",
                           "天地", "奇物", "实力", "火焰", "称号", "地方", "汉火"]
    original_names = [n for n in name_candidates if 2 <= len(n) <= 6 and not any(kw in n for kw in non_person_keywords)]
    missing_names = [n for n in original_names if not any(n in ex or ex in n for ex in existing_names)]

    if not missing_names:
        return {"missing": [], "supplement_count": 0}

    # 确定性补漏: 每个漏名取1句最短相关句，紧凑化，合入E表
    sentences = re.split(r'[。！？\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    supplement_entities = []
    for name in missing_names:
        relevant = [s for s in sentences if name in s]
        if not relevant:
            supplement_entities.append({"n": name, "A": ["原文提及"]})
            continue
        # 合并所有相关句的紧凑化属性(不再只取最短1句，避免空壳实体)
        all_compact = []
        for rel_sent in relevant[:3]:  # 最多取3句
            cs = compactify(rel_sent)
            parts = [a.strip() for a in re.split(r'[，,；;→]', cs) if len(a.strip()) > 1 and name not in a]
            all_compact.extend(parts)
        # 去重 + 分离属性/关系
        seen = set()
        attrs = []
        rels = []
        rel_keywords = ["朋友", "师徒", "敌人", "弟子", "师父", "徒弟", "对手", "盟友",
                         "叛徒", "家主", "女王", "首领", "主人", "属下", "同门"]
        for a in all_compact:
            if a and a not in seen:
                seen.add(a)
                # 关系词放入R而非A
                if any(a == kw or a.startswith(kw) for kw in rel_keywords):
                    rels.append(a)
                else:
                    attrs.append(a)
        if not attrs and not rels:
            # fallback: 最短句整体
            shortest = min(relevant, key=len)
            s = compactify(shortest)
            leftover = s.replace(name, "").strip()
            if leftover:
                attrs = [leftover]
            else:
                attrs = ["原文提及"]
        ent = {"n": name}
        if attrs:
            ent["A"] = attrs
        if rels:
            ent["R"] = rels
        supplement_entities.append(ent)

    return {"missing": missing_names, "supplement_count": len(supplement_entities),
            "supplements": supplement_entities, "supplement_texts": []}


def cooccurrence_scan(text: str, fused_context: dict) -> dict:
    """共现关系扫描: 确定性检测同句共现实体间的关系关键词，补充R字段"""
    if isinstance(fused_context, list):
        entities = fused_context
    else:
        entities = fused_context.get("E", [])

    # 建立名字索引
    name_list = [ent.get("n", "") for ent in entities if ent.get("n")]
    if len(name_list) < 2:
        return {"added_rels": 0}

    # 关系关键词(纯关系词，不含身份称号如女王/家主)
    rel_patterns = {
        "朋友": "朋友", "师徒": "师徒", "敌人": "敌人", "对手": "对手",
        "弟子": "弟子", "师父": "师父", "徒弟": "徒弟", "盟友": "盟友",
        "叛徒": "叛徒",
    }

    # 分句切分：先按句号/感叹号/换行，再按逗号/分号细分(缩小共现窗口)
    raw_sents = re.split(r'[。！？\n]+', text)
    clauses = []
    for s in raw_sents:
        s = s.strip()
        if len(s) <= 5:
            continue
        # 进一步按逗号/分号切分，但保留>=10字的子句
        parts = re.split(r'[，；]', s)
        for p in parts:
            p = p.strip()
            if len(p) >= 10:
                clauses.append(p)
            elif len(s) <= 30:
                # 短句不切，整体作为窗口
                clauses.append(s)
                break

    added = 0
    for clause in clauses:
        # 找出本子句出现的实体
        present = [n for n in name_list if n in clause]
        if len(present) < 2:
            continue
        # 检测关系关键词
        for kw_rel, kw in rel_patterns.items():
            if kw not in clause:
                continue
            # 给离关键词最近的2个实体加关系(分句窗口已缩小，2个足够)
            kw_pos = clause.index(kw)
            scored = [(name, abs(clause.index(name) - kw_pos)) for name in present]
            scored.sort(key=lambda x: x[1])
            for name, _ in scored[:2]:
                for ent in entities:
                    if ent.get("n") == name:
                        rels = normalize_items(ent.get("R", []))
                        if kw_rel not in rels:
                            if "R" not in ent:
                                ent["R"] = []
                            ent["R"].append(kw_rel)
                            added += 1

    return {"added_rels": added}


# ============ Layer 3: 大脑推理 ============
def feedback_search_original(text: str, need_description: str, model: str) -> str:
    """反馈环: 从原文定位缺失信息"""
    prompt = (
        f"在原文中查找关于「{need_description}」的信息。\n"
        f"规则：只输出与查询直接相关的原文片段(原句)，不要改写不要编造。\n"
        f"多句相关则逐条输出。找不到则输出NONE。\n"
        f"原文：{text[:4000]}"
    )
    result = call_ollama(model, prompt, "信息查找。只输出原文相关片段，不改写。")
    if result.strip().upper() in ("NONE", "NONE.", "无", "未找到"):
        return ""
    return result.strip()


def big_brain_answer(fused_context: Dict, question: str, model: str,
                     text: str = "", fuse_brain_model: str = "", max_rounds: int = 2,
                     vram_folding: bool = True, vram_limit: int = 6144,
                     eagle_eye: bool = True, eagle_eye_model: str = "qwen2.5:3b") -> Dict:
    """大脑推理: 读取紧凑行格式机语，回答问题（含VRAM折叠+鹰眼校验）"""
    # 优先使用紧凑格式
    compact = fused_context.get("compact_context", "")
    if compact:
        context_str = compact
    else:
        ctx = fused_context.get("fused_context", fused_context)
        context_str = serialize_compact(ctx) if isinstance(ctx, dict) else json.dumps(ctx, ensure_ascii=False)

    answer = ""
    all_feedback = []
    rounds_done = 0

    for round_num in range(max_rounds + 1):
        if round_num == 0:
            prompt = (
                f"根据机语用自然语言回答。不编造。缺信息写NEED_MORE:缺啥\n"
                f"机语格式:[名] 属性|关系|因果\n"
                f"机语:\n{context_str[:3500]}\n问题:{question}"
            )
        else:
            fb = all_feedback[-1] if all_feedback else ""
            prompt = (
                f"补充信息已到。原机语:\n{context_str[:3000]}\n\n"
                f"补充原文片段:{fb[:1000]}\n问题:{question}"
            )

        system = "推理专家。机语是结构化压缩信息，直接读取回答。缺信息写NEED_MORE:描述。"

        start = time.time()
        is_small_brain = not any(x in model for x in ["14b", "7b", "13b", "v4", "flash", "deepseek"])
        raw = vram_aware_call(model, prompt, system, timeout=240,
                              vram_folding=vram_folding, vram_limit=vram_limit,
                              keep_alive=3600 if is_small_brain else None)
        elapsed = time.time() - start

        if "NEED_MORE:" in raw and round_num < max_rounds and text:
            need = raw.split("NEED_MORE:")[-1].strip()[:200]
            print(f"  [反馈] 第{round_num+1}轮: {need}")
            # v1.7.0: 原文短(<500字)时直接注入全文，不走搜索
            if len(text) < 500:
                all_feedback.append(text)
            else:
                supp = feedback_search_original(text, need, fuse_brain_model)
                all_feedback.append(supp if supp else "[原文中未找到相关信息]")
            rounds_done += 1
        else:
            answer = raw
            break

    if not answer:
        answer = raw

    # Layer 3.5: 鹰眼校验 — 懒加载: 高置信度跳过 + NEED_MORE强制校验
    ee_result = None
    retry_result = None
    if eagle_eye and answer:
        # 如果答案包含NEED_MORE且有原文，强制走鹰眼重提取
        if "NEED_MORE" in answer and text:
            print(f"  [鹰眼] NEED_MORE检测，强制原文重读...", end="", flush=True)
            ee_result = {"verdict": "NEED_REVIEW", "issue": "答案含NEED_MORE",
                         "source_fragment": text[:500], "focus_area": "重新阅读全文"}
            print(f" 红圈! 答案含NEED_MORE...")
            print(f"  [红圈] 定位原文: {text[:80]}...")
            print(f"  [红圈] 关注点: 全文关键信息")
            
            if vram_folding and any(x in model for x in ["14b", "7b", "13b"]):
                unload_model(model)
                time.sleep(1)
            
            retry_result = eagle_eye_retry(
                question, context_str,
                text[:500], "全文关键信息",
                original_text=text,
                extractor_model="qwen2.5:1.5b",
                brain_model=model,
                vram_folding=vram_folding
            )
            answer = retry_result["final_answer"]
        else:
            # 懒鹰眼: 快速判断是否需要校验
            need_eagle = True
            # 启发式: 短答案+明确匹配提取内容+无NEED_MORE → 大概率正确
            if (len(answer) < 80 and "NEED_MORE" not in answer
                and "不确定" not in answer and "无法" not in answer
                and "没有" not in answer[:30]):
                entities_in_context = sum(1 for e in fused_context.get("E", []) if e.get("n"))
                if entities_in_context >= 2:
                    context_ents = " ".join(e.get("n","") for e in fused_context.get("E", [])[:3])
                    if any(name in answer for name in context_ents.split() if len(name) >= 2):
                        need_eagle = False
            
            if need_eagle:
                print(f"  [鹰眼] 审视中...", end="", flush=True)
                ee_result = eagle_eye_validate(question, answer, context_str,
                                                original_text=text, model=eagle_eye_model,
                                                vram_folding=vram_folding)
                if ee_result["verdict"] == "NEED_REVIEW":
                    print(f" 红圈! {ee_result['issue'][:60]}...")
                    print(f"  [红圈] 定位原文: {ee_result['source_fragment'][:80]}...")
                    print(f"  [红圈] 关注点: {ee_result['focus_area']}")
                    
                    if vram_folding and any(x in model for x in ["14b", "7b", "13b"]):
                        unload_model(model)
                        time.sleep(1)
                    
                    retry_result = eagle_eye_retry(
                        question, context_str,
                        ee_result["source_fragment"], ee_result["focus_area"],
                        original_text=text,
                        extractor_model="qwen2.5:1.5b",
                        brain_model=model,
                        vram_folding=vram_folding
                    )
                    answer = retry_result["final_answer"]
                else:
                    print(f" {ee_result['verdict']}")
            else:
                print(f"  [鹰眼] 跳过(高置信度)")
                ee_result = {"verdict": "SKIPPED", "issue": "lazy_eagle", "reason": "high_confidence"}
    elif eagle_eye and not answer:
        ee_result = {"verdict": "SKIPPED", "issue": "no_answer"}

    return {
        "answer": answer,
        "feedback_rounds": rounds_done,
        "feedback_details": all_feedback,
        "time": round(elapsed, 1),
        "eagle_eye": ee_result,
        "eagle_retry": retry_result
    }


# ============ Layer 3.5: 鹰眼校验 (老师红笔模式) ============
def eagle_eye_validate(question: str, answer: str, compact_context: str,
                       original_text: str = "", model: str = "qwen2.5:3b",
                       vram_folding: bool = True) -> Dict:
    """鹰眼校验: 老师红笔模式 — 只标红圈找问题，不做修正
    
    校验维度:
    1. 事实一致性: 答案与机语上下文是否矛盾
    2. 逻辑自洽性: 推理是否自圆其说
    3. 信息充分性: 是否遗漏关键信息
    4. 歧义检测: 表述是否模糊
    
    返回: {verdict, issue, source_fragment, focus_area, time}
    verdict: CORRECT(放行) / NEED_REVIEW(标红圈,触发小脑重学)
    """
    if len(answer.strip()) < 5:
        return {"verdict": "CORRECT", "issue": "", "source_fragment": "", "focus_area": "", "time": 0}
    
    prompt = f"""鹰眼=红笔教师。模糊扫一眼答案vs机语vs原文，发现不一致就标红圈。
规则:不一致→标NEED_REVIEW|原句复制(不改写)|指出关注点|完全正确才CORRECT

问题:{question}
机语:{compact_context[:2000]}
答案:{answer}
原文:{original_text[:2000]}

四行格式:
判定: CORRECT / NEED_REVIEW
红圈: (一句话)
课本段落: (原句复制)
眼睛看哪: (关注实体/属性)"""

    system = "鹰眼=红笔。只标错不修。原句复制。四行格式。"

    start = time.time()
    raw = call_ollama(model, prompt, system, timeout=120)
    elapsed = round(time.time() - start, 1)
    
    verdict = "CORRECT"
    issue = ""
    source_fragment = ""
    focus_area = ""
    
    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith("判定:") or line.startswith("判定："):
            v = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
            verdict = v.upper()
        elif line.startswith("红圈:") or line.startswith("红圈："):
            issue = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
        elif line.startswith("课本段落:") or line.startswith("课本段落："):
            source_fragment = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
        elif line.startswith("眼睛看哪:") or line.startswith("眼睛看哪："):
            focus_area = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
    
    return {
        "verdict": verdict,
        "issue": issue,
        "source_fragment": source_fragment,
        "focus_area": focus_area,
        "time": elapsed,
        "raw": raw[:200]
    }


def eagle_eye_retry(question: str, compact_context: str, source_fragment: str,
                    focus_area: str, original_text: str = "",
                    extractor_model: str = "qwen2.5:1.5b",
                    brain_model: str = "qwen2.5:3b",
                    vram_folding: bool = True) -> Dict:
    """鹰眼红圈→直接注入原文→大脑重答
    
    v2.0优化: 跳过不可靠的1.5B重提取，直接把原文片段注入机语
    让大脑直接读原文重新回答，更快更准
    """
    start = time.time()
    
    # 直接追加原文片段到机语（跳过1.5B重提取）
    print(f"  [红圈→原文] 直接注入原文片段(focus={focus_area[:40]})...")
    
    # 构建包含原文的上下文
    updated_context = compact_context + f"\n\n[红圈原文]: {source_fragment[:500]}"
    
    # 让大脑直接读原文重新回答
    print(f"  [红圈→大脑] 3B重读原文回答...", end="", flush=True)
    
    brain_prompt = (
        f"老师圈了课本关键段落，重读后回答。不编造。\n"
        f"问题:{question}\n"
        f"原笔记:{compact_context[:1500]}\n"
        f"[红圈原文]:{source_fragment[:800]}"
    )
    
    start2 = time.time()
    raw_brain = call_ollama(brain_model, brain_prompt,
                             "重读红圈原文后直接回答。不编造。",
                             timeout=90)
    brain_time = round(time.time() - start2, 1)
    
    print(f" {brain_time}s")
    
    return {
        "retry_done": True,
        "new_entities": 0,  # 不再重提取
        "new_lines": 1,     # 追加了原文
        "extract_time": 0,  # 跳过提取
        "brain_time": brain_time,
        "updated_context": updated_context,
        "final_answer": raw_brain,
        "total_retry_time": round(time.time() - start, 1)
    }


# ============ 内置测试 ============
TEST_TEXT = (
    "萧炎抬头望向那巨大的黑角域方向，眼中有着一丝凝重。"
    "在黑角域之中，有着不少的强者，其中最为著名的便是韩枫，"
    "此人乃是药老的叛徒弟子，当年偷袭药老导致其灵魂体被迫遁入骨灵冷火之中。"
    "韩枫如今已是斗皇强者，掌控着黑角域最大的势力枫城。\n\n"
    "药老沉声道：\u201c韩枫，我那不肖弟子，当年若非他偷袭，我也不至于落到这般田地。\u201d"
    "萧炎闻言，心中对韩枫的恨意更甚，拳头紧握。"
    "药老是他最尊敬的师父，韩枫对药老的所作所为，他绝不会轻易放过。\n\n"
    "萧战望着远去的萧炎背影，心中百感交集。"
    "他这个儿子，从小便展现出过人的天赋，却又经历了三年废物的屈辱。"
    "如今能重新崛起，全靠自身的坚毅。"
    "萧战身为乌坦城萧家家主，虽然实力不过大斗师级别，但多年来将萧家治理得井井有条。\n\n"
    "萧炎进入迦南学院修炼，在这里遇到了不少强敌。"
    "迦南学院后山有着一座天焚炼气塔，塔底封印着陨落心炎，这是一种异火排名第十四的天地奇物。"
    "萧炎在炼气塔中修炼，不仅实力突飞猛进，更与美杜莎女王产生了纠葛。"
    "美杜莎是蛇人族的女王，拥有极其强大的实力，乃是斗宗级别的强者。\n\n"
    "小医仙是萧炎在魔兽山脉结识的朋友，她天生厄难毒体，"
    "这种特殊体质使得她不修炼也能自动吸收天地毒素提升实力，"
    "但同时也意味着她随时可能被毒素反噬失控。"
    "萧炎曾承诺会帮她控制毒体，这份承诺一直记在心中。"
)

TEST_QUESTIONS = [
    "药老的叛徒弟子是谁？他做了什么？",
    "萧炎和小医仙是什么关系？小医仙有什么特殊体质？",
    "美杜莎是谁？她的实力等级是什么？",
]


def run_test():
    print("=" * 60)
    print("zhongxing 1.6.2 内置测试 (分块+降噪+VRAM折叠)")
    print("=" * 60)

    config = FALLBACK_CONFIG
    vram_folding = config.get("vram_folding", True)
    vram_limit = config.get("vram_limit_mb", 6144)
    eagle_eye_enabled = config.get("eagle_eye_enabled", True)
    eagle_eye_model = config.get("eagle_eye_model", "qwen2.5:3b")
    
    used, free, total = nvidia_smi()
    if used:
        print(f"GPU: {used}/{total} MiB ({free} MiB free) | VRAM折叠: {'ON' if vram_folding else 'OFF'} | 鹰眼: {'ON' if eagle_eye_enabled else 'OFF'}")
    print(f"文本:{len(TEST_TEXT)}字 | {len(TEST_QUESTIONS)}题\n")

    total_start = time.time()

    # Layer 0: 维度自发现
    print("[1/4] 维度自发现...")
    schema = discover_schema(TEST_TEXT, config["schema_model"])
    print(f"  类型: {schema['text_type']}")
    print(f"  实体: {', '.join(schema['entity_types'][:5])}")
    print(f"  属性: {', '.join(schema['attr_types'][:5])}")
    print(f"  关系: {', '.join(schema['rel_types'][:5])}")

    # Layer 1: 通用提取(含自动分块+Pipeline重试)
    print(f"\n[2/4] 通用提取({config['num_extractors']}个并行)...")
    extractions = run_extractors(TEST_TEXT, schema, config["extractor_model"], config["num_extractors"])
    total_ents = sum(len(e["entities"]) for e in extractions)
    if total_ents < 5:
        print(f"  总实体{total_ents}<5，重跑提取...")
        extractions2 = run_extractors(TEST_TEXT, schema, config["extractor_model"], config["num_extractors"])
        total_ents2 = sum(len(e["entities"]) for e in extractions2)
        # 取实体更多的那次
        if total_ents2 > total_ents:
            extractions = extractions2
            print(f"  重跑更好: {total_ents2}实体 > {total_ents}实体")

    # Layer 2: RRF融合
    print("\n[3/4] RRF融合(确定性紧凑化)...")
    fusion = rrf_fuse(extractions, config["schema_model"], len(TEST_TEXT))
    if isinstance(fusion["fused_context"], list):
        fusion["fused_context"] = {"E": fusion["fused_context"]}
    ent_count = len(fusion["fused_context"].get("E", []))
    print(f"  耗时:{fusion['time']}s, {ent_count}实体")
    print(f"  JSON:{fusion['json_len']}字 → 紧凑:{fusion['compact_len']}字")
    if fusion.get("rrf_ranking"):
        top3 = [(n, f"{s:.3f}", c) for n, s, c in fusion["rrf_ranking"][:5]]
        print(f"  RRF Top5: {top3}")

    # 补漏校验
    print("\n  人名校验(补漏)...")
    gap = entity_gap_check(TEST_TEXT, fusion["fused_context"], config["schema_model"])
    if gap["missing"]:
        print(f"  漏掉: {','.join(gap['missing'])} → 补入{gap['supplement_count']}个")
        existing_E = fusion["fused_context"].get("E", [])
        if isinstance(existing_E, list):
            existing_E.extend(gap.get("supplements", []))
    else:
        print(f"  无遗漏")

    # 共现关系扫描
    cooc = cooccurrence_scan(TEST_TEXT, fusion["fused_context"])
    if cooc["added_rels"] > 0:
        print(f"  共现关系: +{cooc['added_rels']}条")

    # 重新生成紧凑格式
    fusion["compact_context"] = serialize_compact(fusion["fused_context"])

    # 压缩比计算
    compact_str = fusion.get("compact_context", serialize_compact(fusion["fused_context"]))
    json_str = json.dumps(fusion["fused_context"], ensure_ascii=False)
    compact_ratio = len(TEST_TEXT) / max(len(compact_str), 1)
    json_ratio = len(TEST_TEXT) / max(len(json_str), 1)
    print(f"\n压缩: {len(TEST_TEXT)}→{len(compact_str)}字 (紧凑{compact_ratio:.1f}:1 | JSON{json_ratio:.1f}:1)")
    print(f"---机语预览---\n{compact_str[:500]}\n---机语结束---")

    # Layer 3: 大脑推理(逐题串行，避免14b内存压力 + 莫比乌斯环VRAM折叠 + 鹰眼校验)
    print(f"\n[4/4] 大脑推理({len(TEST_QUESTIONS)}题串行, VRAM折叠{'ON' if vram_folding else 'OFF'}, 鹰眼{'ON' if eagle_eye_enabled else 'OFF'})...")
    results = []
    for i, q in enumerate(TEST_QUESTIONS):
        pipe = big_brain_answer(fusion, q, config["big_brain_model"],
                                TEST_TEXT, config["schema_model"],
                                config.get("max_feedback_rounds", 2),
                                vram_folding=vram_folding, vram_limit=vram_limit,
                                eagle_eye=eagle_eye_enabled, eagle_eye_model=eagle_eye_model)
        results.append({"i": i, "q": q, "pipeline": pipe})
        fb = f" (反馈{pipe['feedback_rounds']}轮)" if pipe["feedback_rounds"] > 0 else ""
        ee = ""
        if pipe.get("eagle_eye"):
            if pipe["eagle_eye"]["verdict"] == "NEED_REVIEW":
                retry = pipe.get("eagle_retry")
                detail = f"({retry['new_entities']}实体)" if retry else ""
                ee = f" [红圈→重学{detail}]"
            elif pipe["eagle_eye"]["verdict"] != "CORRECT":
                ee = f" [鹰眼:{pipe['eagle_eye']['verdict']}]"
        print(f"  Q{i+1}({pipe['time']}s{fb}{ee}): {pipe['answer'][:150]}")
    
    # 莫比乌斯环: 推理完成后卸载14B释放VRAM
    if vram_folding:
        fold_cleanup([config["big_brain_model"]])
        print("  [VRAM折叠] 14B已卸载，显存已释放")

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"测试汇总 | 紧凑压缩:{compact_ratio:.1f}:1 | JSON:{json_ratio:.1f}:1 | 总耗时:{total_time:.1f}s")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\nQ{i+1}: {r['q']}")
        print(f"  管道: {r['pipeline']['answer'][:200]}")


# ============ 通用测试函数 ============
def run_test_with(text: str, questions: list, config: Dict = None):
    """通用测试，接受任意文本和问题列表（含VRAM折叠）"""
    if config is None:
        config = FALLBACK_CONFIG

    vram_folding = config.get("vram_folding", True)
    vram_limit = config.get("vram_limit_mb", 6144)
    eagle_eye_enabled = config.get("eagle_eye_enabled", True)
    eagle_eye_model = config.get("eagle_eye_model", "qwen2.5:3b")

    print("=" * 60)
    print(f"zhongxing 1.6.3 长文本测试 (分块+降噪+VRAM折叠+鹰眼校验)")
    print("=" * 60)
    
    used, free, total = nvidia_smi()
    if used:
        print(f"GPU: {used}/{total} MiB ({free} MiB free) | VRAM折叠: {'ON' if vram_folding else 'OFF'} | 鹰眼: {'ON' if eagle_eye_enabled else 'OFF'}")

    print(f"文本:{len(text)}字 | {len(questions)}题\n")

    total_start = time.time()

    # Layer 0: 维度自发现
    print("[1/4] 维度自发现...")
    schema = discover_schema(text, config["schema_model"])
    print(f"  类型: {schema['text_type']}")
    print(f"  实体: {', '.join(schema['entity_types'][:5])}")
    print(f"  属性: {', '.join(schema['attr_types'][:5])}")
    print(f"  关系: {', '.join(schema['rel_types'][:5])}")

    # Layer 1: 通用提取(含自动分块+Pipeline重试)
    print(f"\n[2/4] 通用提取({config['num_extractors']}个并行)...")
    extractions = run_extractors(text, schema, config["extractor_model"], config["num_extractors"])
    total_ents = sum(len(e["entities"]) for e in extractions)
    if total_ents < 5:
        print(f"  总实体{total_ents}<5，重跑提取...")
        extractions2 = run_extractors(text, schema, config["extractor_model"], config["num_extractors"])
        total_ents2 = sum(len(e["entities"]) for e in extractions2)
        if total_ents2 > total_ents:
            extractions = extractions2
            print(f"  重跑更好: {total_ents2}实体 > {total_ents}实体")

    # Layer 2: RRF融合
    print("\n[3/4] RRF融合(确定性紧凑化)...")
    fusion = rrf_fuse(extractions, config["schema_model"], len(text))
    if isinstance(fusion["fused_context"], list):
        fusion["fused_context"] = {"E": fusion["fused_context"]}

    ent_count = len(fusion["fused_context"].get("E", []))
    print(f"  耗时:{fusion['time']}s, {ent_count}实体")
    print(f"  JSON:{fusion['json_len']}字 → 紧凑:{fusion['compact_len']}字")
    if fusion.get("rrf_ranking"):
        top3 = [(n, f"{s:.3f}", c) for n, s, c in fusion["rrf_ranking"][:5]]
        print(f"  RRF Top5: {top3}")

    # 补漏校验
    print("\n  人名校验(补漏)...")
    gap = entity_gap_check(text, fusion["fused_context"], config["schema_model"])
    if gap["missing"]:
        print(f"  漏掉: {','.join(gap['missing'])} → 补入{gap['supplement_count']}个")
        existing_E = fusion["fused_context"].get("E", [])
        if isinstance(existing_E, list):
            existing_E.extend(gap.get("supplements", []))
    else:
        print(f"  无遗漏")

    # 共现关系扫描
    cooc = cooccurrence_scan(text, fusion["fused_context"])
    if cooc["added_rels"] > 0:
        print(f"  共现关系: +{cooc['added_rels']}条")

    # 重新生成紧凑格式
    fusion["compact_context"] = serialize_compact(fusion["fused_context"])

    # 压缩比计算
    compact_str = fusion.get("compact_context", serialize_compact(fusion["fused_context"]))
    json_str = json.dumps(fusion["fused_context"], ensure_ascii=False)
    compact_ratio = len(text) / max(len(compact_str), 1)
    json_ratio = len(text) / max(len(json_str), 1)
    print(f"\n压缩: {len(text)}→{len(compact_str)}字 (紧凑{compact_ratio:.1f}:1 | JSON{json_ratio:.1f}:1)")
    print(f"---机语预览(前800字)---\n{compact_str[:800]}\n---机语结束---")

    # Layer 3: 大脑推理(逐题串行，避免14b内存压力 + 莫比乌斯环VRAM折叠 + 鹰眼校验)
    print(f"\n[4/4] 大脑推理({len(questions)}题, VRAM折叠{'ON' if vram_folding else 'OFF'}, 鹰眼{'ON' if eagle_eye_enabled else 'OFF'})...")
    results = []
    for i, q in enumerate(questions):
        pipe = big_brain_answer(fusion, q, config["big_brain_model"],
                                text, config["schema_model"],
                                config.get("max_feedback_rounds", 2),
                                vram_folding=vram_folding, vram_limit=vram_limit,
                                eagle_eye=eagle_eye_enabled, eagle_eye_model=eagle_eye_model)
        results.append({"i": i, "q": q, "pipeline": pipe})
        fb = f" (反馈{pipe['feedback_rounds']}轮)" if pipe["feedback_rounds"] > 0 else ""
        ee = ""
        if pipe.get("eagle_eye"):
            if pipe["eagle_eye"]["verdict"] == "NEED_REVIEW":
                retry = pipe.get("eagle_retry")
                detail = f"({retry['new_entities']}实体)" if retry else ""
                ee = f" [红圈→重学{detail}]"
            elif pipe["eagle_eye"]["verdict"] != "CORRECT":
                ee = f" [鹰眼:{pipe['eagle_eye']['verdict']}]"
        print(f"  Q{i+1}({pipe['time']}s{fb}{ee}): {pipe['answer'][:150]}")
    
    # 莫比乌斯环: 推理完成后卸载14B释放VRAM
    if vram_folding:
        fold_cleanup([config["big_brain_model"]])
        print("  [VRAM折叠] 14B已卸载，显存已释放")

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"测试汇总 | 紧凑压缩:{compact_ratio:.1f}:1 | JSON:{json_ratio:.1f}:1 | 总耗时:{total_time:.1f}s")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\nQ{i+1}: {r['q']}")
        print(f"  管道: {r['pipeline']['answer'][:300]}")


# ============ 交互模式 ============
def interactive_mode(config: Dict = None):
    if config is None:
        config = FALLBACK_CONFIG

    vram_folding = config.get("vram_folding", True)
    vram_limit = config.get("vram_limit_mb", 6144)
    eagle_eye_enabled = config.get("eagle_eye_enabled", True)
    eagle_eye_model = config.get("eagle_eye_model", "qwen2.5:3b")

    print("\n" + "=" * 60)
    print("zhongxing 1.6.3 - 交互模式 (分块+降噪+VRAM折叠+鹰眼校验)")
    print("=" * 60)
    print("输入文本 → 预处理(压缩) → 反复提问 | :quit退出 | :ctx看机语")
    
    used, free, total = nvidia_smi()
    if used:
        print(f"GPU: {used}/{total} MiB | VRAM折叠: {'ON' if vram_folding else 'OFF'} | 鹰眼: {'ON' if eagle_eye_enabled else 'OFF'}")
    print()

    print("请输入文本（:done结束）:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ":done":
            break
        lines.append(line)
    text = "\n".join(lines)
    if not text.strip():
        print("[ERROR] 没有输入文本")
        return

    print("\n[预处理] 维度自发现...")
    schema = discover_schema(text, config["schema_model"])
    print(f"  类型: {schema['text_type']}, 实体: {', '.join(schema['entity_types'][:3])}")

    print(f"[预处理] 通用提取({config['num_extractors']}个并行)...")
    extractions = run_extractors(text, schema, config["extractor_model"], config["num_extractors"])
    total_ents = sum(len(e["entities"]) for e in extractions)
    if total_ents < 5:
        print(f"  总实体{total_ents}<5，重跑提取...")
        extractions2 = run_extractors(text, schema, config["extractor_model"], config["num_extractors"])
        total_ents2 = sum(len(e["entities"]) for e in extractions2)
        if total_ents2 > total_ents:
            extractions = extractions2

    print("[预处理] RRF融合(确定性紧凑化)...")
    fusion = rrf_fuse(extractions, config["schema_model"], len(text))
    if isinstance(fusion["fused_context"], list):
        fusion["fused_context"] = {"E": fusion["fused_context"]}

    print("[预处理] 人名校验...")
    gap = entity_gap_check(text, fusion["fused_context"], config["schema_model"])
    if gap["missing"]:
        existing_E = fusion["fused_context"].get("E", [])
        if isinstance(existing_E, list):
            existing_E.extend(gap.get("supplements", []))
        print(f"  补入: {','.join(gap['missing'])}")

    # 共现关系扫描
    cooc = cooccurrence_scan(text, fusion["fused_context"])
    if cooc["added_rels"] > 0:
        print(f"  共现关系: +{cooc['added_rels']}条")

    fusion["compact_context"] = serialize_compact(fusion["fused_context"])

    compact_str = fusion.get("compact_context", serialize_compact(fusion["fused_context"]))
    ratio = len(text) / max(len(compact_str), 1)
    print(f"\n预处理完成! {len(text)}→{len(compact_str)}字 ({ratio:.1f}:1)\n")

    while True:
        try:
            q = input("问题(:quit/:ctx): ").strip()
        except EOFError:
            break
        if not q or q == ":quit":
            break
        if q == ":ctx":
            print(compact_str[:2000])
            continue
        print("推理中...")
        a = big_brain_answer(fusion, q, config["big_brain_model"],
                             text=text, fuse_brain_model=config["schema_model"],
                             max_rounds=config.get("max_feedback_rounds", 2),
                             vram_folding=vram_folding, vram_limit=vram_limit,
                             eagle_eye=eagle_eye_enabled, eagle_eye_model=eagle_eye_model)
        fb = f" (反馈{a['feedback_rounds']}轮)" if a["feedback_rounds"] > 0 else ""
        ee = ""
        if a.get("eagle_eye") and a["eagle_eye"]["verdict"] not in ("CORRECT",):
            ee = f" [鹰眼:{a['eagle_eye']['verdict']}]"
        print(f"\n({a['time']}s{fb}{ee}) {a['answer']}\n")
        # 莫比乌斯环: 每问后卸载14B释放VRAM
        if vram_folding:
            fold_cleanup([config["big_brain_model"]])


# ============ 自动检测配置 ============
def auto_detect_config(available_models: list) -> Dict:
    config = dict(FALLBACK_CONFIG)

    # 提取器: 优先qwen2.5:1.5b(结构化输出稳定)
    for m in available_models:
        if "qwen2.5:1.5b" in m.lower():
            config["extractor_model"] = m; break
    else:
        for m in available_models:
            if "minicpm5" in m.lower():
                config["extractor_model"] = m; break

    # schema模型(NER+维度发现): 优先3b(列人名更准)
    for m in available_models:
        if "qwen2.5:3b" in m.lower():
            config["schema_model"] = m; break
    else:
        for m in available_models:
            if "qwen2.5:1.5b" in m.lower():
                config["schema_model"] = m; break

    # 大脑: 最大的
    for candidate in ["qwen2.5:14b", "qwen3.6", "qwen2.5:7b"]:
        for m in available_models:
            if candidate in m.lower():
                config["big_brain_model"] = m; return config

    return config


# ============ 入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="zhongxing 1.6.3 - 通用压缩管道(分块+降噪+VRAM折叠+鹰眼校验)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--file", type=str, default="", help="外部测试文件(.py，需含LONG_TEXT和LONG_QUESTIONS)")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--extractor", type=str, default="")
    parser.add_argument("--schema-model", type=str, default="")
    parser.add_argument("--big-brain", type=str, default="")
    parser.add_argument("--eagle-eye-model", type=str, default="", help="鹰眼校验模型(默认qwen2.5:3b)")
    parser.add_argument("--num-extractors", type=int, default=3)
    parser.add_argument("--max-feedback", type=int, default=2)
    parser.add_argument("--no-vram-folding", action="store_true", help="禁用VRAM折叠(莫比乌斯环模式)")
    parser.add_argument("--vram-limit", type=int, default=6144, help="VRAM目标上限MiB(默认6144=6GB)")
    parser.add_argument("--no-eagle-eye", action="store_true", help="禁用鹰眼校验")

    args = parser.parse_args()

    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            available_models = [m["name"] for m in json.loads(r.read().decode("utf-8")).get("models", [])]
        print(f"Ollama已连接: {', '.join(available_models)}")
    except:
        print("[ERROR] Ollama未启动，请先运行: ollama serve")
        sys.exit(1)

    config = auto_detect_config(available_models)

    if args.extractor: config["extractor_model"] = args.extractor
    if args.schema_model: config["schema_model"] = args.schema_model
    if args.big_brain: config["big_brain_model"] = args.big_brain
    config["num_extractors"] = args.num_extractors
    config["max_feedback_rounds"] = args.max_feedback

    # VRAM折叠配置
    if args.no_vram_folding:
        config["vram_folding"] = False
    config["vram_limit_mb"] = args.vram_limit

    # 鹰眼校验配置
    if args.no_eagle_eye:
        config["eagle_eye_enabled"] = False
    if args.eagle_eye_model:
        config["eagle_eye_model"] = args.eagle_eye_model

    needed = {config["extractor_model"], config["schema_model"], config["big_brain_model"]}
    missing = needed - set(available_models)
    if missing:
        print(f"\n[!] 缺少模型: {missing}")
        print(f"    需要: ollama pull qwen2.5:1.5b qwen2.5:14b")
        sys.exit(1)

    print(f"\n当前配置:")
    print(f"  提取器({config['num_extractors']}个): {config['extractor_model']} [通用并行提取]")
    print(f"  NER+维度: {config['schema_model']} [人名识别+维度发现]")
    print(f"  大脑:      {config['big_brain_model']} [推理]")
    print(f"  鹰眼:      {config['eagle_eye_model']} [校验] {'ON' if config.get('eagle_eye_enabled', True) else 'OFF'}")
    print(f"  反馈环:    ≤{config['max_feedback_rounds']}轮")
    print(f"  序列化:    紧凑行格式 (v1.5)")
    vram_status = "OFF" if args.no_vram_folding else f"ON({config['vram_limit_mb']}MiB)"
    print(f"  VRAM折叠:  {vram_status} [莫比乌斯环模式]")

    if args.file:
        # 外部文件模式
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_data", args.file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        test_text = mod.LONG_TEXT
        test_questions = mod.LONG_QUESTIONS
        run_test_with(test_text, test_questions, config)
    elif args.test:
        run_test()
    elif args.interactive:
        interactive_mode(config)
    else:
        run_test()
