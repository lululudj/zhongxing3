# -*- coding: utf-8 -*-
"""
众星管道 V4-Flash集成测试
- 验证284B DeepSeek V4-Flash作为大脑模型
- 1.5B/3B仍使用Ollama (提取+校验)
- 测试推理速度是否达到15秒目标
"""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from zhongxing_agent import (
    discover_schema, run_extractors, rrf_fuse,
    big_brain_answer, FALLBACK_CONFIG
)

CONFIG = dict(FALLBACK_CONFIG)
CONFIG["big_brain_model"] = "deepseek-v4-flash"
CONFIG["eagle_eye_enabled"] = False
CONFIG["vram_folding"] = False

LONG_TEXT = (
    "萧炎，萧家三少爷，三年前是乌坦城第一天才，十岁凝聚斗之气旋，"
    "震惊全城。然而三年前神秘失踪，回来后斗气全失，沦为废材，"
    "被全城嘲笑。纳兰嫣然是云岚宗少宗主，与萧炎有婚约在身，"
    "却因萧炎沦为废材而上门退婚。萧炎愤而立下三年之约，"
    "约定三年后上云岚宗挑战。药老是藏身于萧炎戒指中的神秘灵魂，"
    "曾是大陆第一炼药师，为了收萧炎为徒，帮助他重新修炼。"
    "萧薰儿是古族大小姐，从小与萧炎青梅竹马，一直默默守护在他身边，"
    "拥有极强的实力。加玛帝国位于斗气大陆西北，实力最强者为斗皇巅峰。"
    "萧炎父亲萧战是萧家族长，一直以萧炎为傲。"
)

LONG_QUESTIONS = [
    "萧炎为什么要立下三年之约？",
]

def run_test():
    print("=" * 60)
    print("众星管道 V4-Flash 集成测试")
    print(f"大脑: {CONFIG['big_brain_model']} (284B CPU/mmap)")
    print(f"提取器: {CONFIG['extractor_model']} (Ollama)")
    print(f"维度/NER: {CONFIG['schema_model']} (Ollama)")
    print("=" * 60)

    total_start = time.time()

    for qi, question in enumerate(LONG_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"问题 {qi+1}: {question}")
        print(f"{'='*60}")

        # Stage 0: 维度发现
        t0 = time.time()
        schema = discover_schema(LONG_TEXT[:2000], CONFIG["schema_model"])
        print(f"  [Layer0 维度] {time.time()-t0:.1f}s")

        # Stage 1: 并行提取
        t1 = time.time()
        extracted = run_extractors(
            LONG_TEXT, schema, CONFIG["extractor_model"],
            num=CONFIG["num_extractors"]
        )
        print(f"  [Layer1 提取] {time.time()-t1:.1f}s, {len(extracted)}实体")

        # Stage 2: RRF融合+压缩
        t2 = time.time()
        fused = rrf_fuse(extracted, CONFIG["schema_model"], len(LONG_TEXT))
        compact = fused.get("compact_context", "")
        print(f"  [Layer2 融合] {time.time()-t2:.1f}s, 压缩{len(compact)}字")

        # Stage 3: 大脑推理(V4-Flash)
        print(f"  [Layer3 大脑] V4-Flash子进程中...(284B CPU,预计较慢)")
        t3 = time.time()
        result = big_brain_answer(
            fused, question, CONFIG["big_brain_model"],
            text=LONG_TEXT,
            max_rounds=1,
            vram_folding=False,
            eagle_eye=False,
        )
        brain_time = time.time() - t3
        answer = result.get("answer", str(result))
        fb_rounds = result.get("feedback_rounds", 0)
        print(f"  [Layer3 大脑] {brain_time:.1f}s (反馈{fb_rounds}轮)")
        print(f"  答案: {answer[:500]}")

        total_time = time.time() - total_start
        print(f"\n  总耗时: {total_time:.1f}s")

    print(f"\n测试完成")

if __name__ == "__main__":
    run_test()
