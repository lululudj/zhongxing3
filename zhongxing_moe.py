# -*- coding: utf-8 -*-
"""
众星系统 MoE 大脑集成
支持 DeepSeek V3 / R1 等 MoE 模型作为大脑
"""

import urllib.request
import json
import time

OLLAMA_BASE = "http://127.0.0.1:11434"


def call_moe_brain(prompt: str, model: str = "deepseek-r1:7b",
                   context: str = "", timeout: int = 300) -> str:
    """
    调用MoE模型作为大脑推理

    Args:
        prompt: 问题或机语
        model: MoE模型名称
        context: 上下文（机语）
        timeout: 超时时间

    Returns:
        模型回答
    """
    system = """你是众星系统的大脑，负责根据机语进行推理回答。
机语格式: [名] 属性|关系|因果
规则：
1. 直接根据机语回答，不编造
2. 缺少信息时写 NEED_MORE: 缺什么
3. 简洁准确，不废话"""

    messages = []
    if context:
        messages.append({"role": "system", "content": f"机语:\n{context}"})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_gpu": 35,  # GPU层数，根据显存调整
        }
    }

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
        return data["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def zhongxing_moe_pipeline(text: str, question: str,
                           extractor_model: str = "qwen2.5:1.5b",
                           needle_model: str = "qwen2.5:3b",
                           brain_model: str = "deepseek-r1:7b",
                           eagle_model: str = "qwen2.5:3b"):
    """
    众星管道 - MoE大脑版

    流程：
    1. 1.5B小脑提取机语
    2. 3B纳米针压缩精炼
    3. MoE大脑推理
    4. 3B鹰眼校验
    5. 循环修正
    """
    from zhongxing_agent import (
        universal_extract, serialize_compact,
        eagle_eye_validate, eagle_eye_retry
    )

    t0 = time.time()

    # Schema配置
    schema = {
        "text_type": "通用",
        "entity_types": ["人物", "机构", "概念", "事物", "地点", "时间"],
        "attr_types": ["属性", "数值", "时间"],
        "rel_types": ["关系", "因果"]
    }

    print(f"[小脑] 1.5B提取机语...")
    ext = universal_extract(text, 0, schema, extractor_model, question=question)
    ents = ext.get("entities", [])
    compact = serialize_compact({"E": ents})
    print(f"  机语长度: {len(compact)}字")

    print(f"[纳米针] 3B压缩精炼...")
    # 这里可以加入3B压缩步骤，暂时跳过

    print(f"[大脑] MoE推理 ({brain_model})...")
    answer = call_moe_brain(question, model=brain_model, context=compact)
    print(f"  答案: {answer[:100]}...")

    # 检查NEED_MORE
    if "NEED_MORE:" in answer:
        need = answer.split("NEED_MORE:")[-1].strip()[:200]
        print(f"  [反馈] 需要更多信息: {need}")
        # 可以加入搜索原文的逻辑

    print(f"[鹰眼] 3B校验...")
    ee = eagle_eye_validate(question, answer, compact,
                           original_text=text, model=eagle_model)

    if ee.get("verdict") == "NEED_REVIEW":
        print(f"  [红圈] {ee.get('issue', '')[:60]}")
        # 重提取 + 重推理
        retry = eagle_eye_retry(
            question, compact,
            ee.get("source_fragment", ""),
            ee.get("focus_area", ""),
            original_text=text,
            extractor_model=extractor_model,
            brain_model=brain_model
        )
        answer = retry.get("final_answer", answer)
    else:
        print(f"  [通过] 答案正确")

    elapsed = round(time.time() - t0, 1)
    print(f"\n总耗时: {elapsed}s")

    return {
        "answer": answer,
        "time": elapsed,
        "eagle_eye": ee.get("verdict", "N/A"),
        "compact_length": len(compact)
    }


if __name__ == "__main__":
    # 测试
    text = """
    牛顿第二定律指出，物体的加速度与作用力成正比，与质量成反比，公式为F=ma。
    这个定律是经典力学的基础，由艾萨克·牛顿在1687年提出。
    """

    question = "牛顿第二定律的公式是什么？谁提出的？"

    result = zhongxing_moe_pipeline(text, question)
    print(f"\n最终答案: {result['answer']}")
    print(f"耗时: {result['time']}s")
