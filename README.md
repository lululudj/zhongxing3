# 众星3.0 — 把不可能变为可能

> **在32GB内存的笔记本上，跑起了284B参数的DeepSeek V4-Flash。**

---

## 这件事为什么"不可能"

DeepSeek V4-Flash 是一个 **2840亿参数** 的混合专家模型（MoE），即使经过最低精度的IQ2_XXS量化，模型文件仍然高达 **86.7GB**。

而我的电脑：

| 硬件 | 配置 |
|------|------|
| CPU | Intel i7-13620H（16线程） |
| GPU | NVIDIA RTX 4060 Laptop（8GB显存） |
| 内存 | 32GB DDR4 |
| 硬盘 | 普通NVMe SSD |

所有人都说：**32GB内存跑86GB的模型？不可能。** 8GB显存连模型的一个零头都装不下。主流推理引擎（Ollama、LM Studio、官方llama.cpp）根本不支持deepseek4架构。

---

## 怎么做到的

### 第一关：架构支持

官方llama.cpp不支持deepseek4架构。antirez（Redis作者）做了一个fork，专门添加了deepseek4支持：

```
https://github.com/antirez/llama.cpp-deepseek-v4-flash
```

这是**唯一**能跑V4-Flash的开源推理引擎。我从源码编译了它。

### 第二关：内存不够

86GB模型 vs 32GB内存，差了将近3倍。解决方案是 **mmap（内存映射文件）**：

- 不把整个模型加载到内存
- 而是把模型文件映射到虚拟地址空间
- 只在需要时才读取对应的部分到物理内存
- V4-Flash是MoE架构，每次推理只激活6/256个专家（约13B参数）
- 实际内存占用远小于86GB

**结果：86GB模型成功在32GB内存上加载并生成文本。**

### 第三关：众星管道

光能跑还不够，我设计了"众星管道"让小模型和大模型协作：

```
用户问题
  → 3B维度发现（快速提取关键信息）
  → 1.5B并行提取（多角度抽取）
  → RRF融合压缩（合并去重）
  → V4-Flash大脑推理（深度思考）
  → 3B鹰眼校验（质量把关）
  → 最终回答
```

管道前3层（发现+提取+压缩）仅需 **5.4秒**，把冗长输入压缩成精华，大幅减少V4-Flash需要处理的内容量。

---

## 实测结果

| 指标 | 结果 |
|------|------|
| 模型加载 | 成功（mmap，约2分钟） |
| 文本生成 | 成功（纯CPU，1-3 tok/s） |
| 众星管道前3层 | 5.4秒 |
| 端到端回答 | 可用（慢但正确） |

**是的，1-3 tok/s很慢。但这不是重点。**

重点是：一个所有人说"不可能"的事，**做成了**。

---

## 为什么这件事有意义

### 1. 证明了mmap的威力

mmap让"模型必须小于内存"这个常识被打破。86GB > 32GB，但mmap让它跑起来了。这意味着未来更大的模型，只要你的硬盘放得下，就有可能在低配设备上运行。

### 2. 证明了MoE的友好性

MoE架构每次只激活一小部分专家，这意味着mmap的缺页中断不会太频繁——大部分时候你只需要读取一小部分权重。稠密模型就没这么幸运了。

### 3. 众星管道的思路

让小模型做粗活，大模型只做关键决策。这和人类组织很像——实习生做调研，总监做决策。即使总监很慢，整体效率仍然远超总监一个人从头做到尾。

---

## 诚实地说

| 做到了 | 没做到 |
|--------|--------|
| 86GB模型在32GB内存上跑起来 | 速度不够快（1-3 tok/s） |
| 众星管道架构验证通过 | GPU加速受限于8GB显存 |
| 端到端推理成功 | 推测解码因内存不足无法启用 |
| 完整编译链路打通 | 20 tok/s目标未达成 |

**速度是下一个要攻克的难题。** 可能的方向：
- KTransformers：热门专家放GPU，冷门专家放CPU
- 在线蒸馏：让小模型从V4-Flash的回答中学习，逐渐替代
- 换更小但能力接近的MoE模型

---

## 技术细节

### 编译命令

```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### 运行命令

```bash
llama-cli.exe -m deepseek-v4-flash-iq2xxs.gguf -p "你好" -n 100 -c 1024 -t 4
```

### 众星管道

```python
# 众星管道核心流程
def zhongxing_pipeline(question):
    schema = discover_schema(question)      # 3B维度发现
    extracted = run_extractors(question, schema)  # 1.5B并行提取
    compressed = rrf_fuse(extracted)        # RRF融合压缩
    answer = big_brain_answer(compressed)   # V4-Flash推理
    verified = eagle_eye_check(answer)      # 3B鹰眼校验
    return verified
```

---

## 项目链接

- 众星2.0代码仓库：[lululudj/zhongxing2](https://github.com/lululudj/zhongxing2)
- 推理引擎：[antirez/llama.cpp-deepseek-v4-flash](https://github.com/antirez/llama.cpp-deepseek-v4-flash)
- DeepSeek V4-Flash模型：[HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

---

## 写在最后

我是一个软件工程新人。这个项目从零开始：学git、学编译、学模型架构、学推理引擎、写管道代码、调bug、测速度。每一步都在踩坑，每一步都在查资料。

但最终，我在32GB内存的笔记本上跑起了284B参数的DeepSeek V4-Flash。

**有人说这不可能。我说，试了才知道。**

---

*众星3.0 · 2026.06.16*
