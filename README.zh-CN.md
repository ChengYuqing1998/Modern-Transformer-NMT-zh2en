[English](README.md) | [中文](README.zh-CN.md)

# Modern-Transformer-NMT-zh2en

`Modern-Transformer-NMT-zh2en` 是一个纯 PyTorch 的教育型机器翻译项目，最初基于经典的 encoder-decoder Transformer 实现。

目前仓库包含两条相关的模型路径：

- 经典 encoder-decoder Transformer，并提供一个可直接运行中译英推理的预训练 checkpoint；
- 用于实验的 decoder-only 翻译模型，可以通过配置开关比较经典 Transformer 组件与现代 LLM 组件。

项目使用 `cn-eng.txt` 中约 90,000 组中英文句对。注意力、mask、训练目标、生成循环和架构消融都直接实现在仓库中，便于阅读、修改和训练。

## 为什么有两种架构？

原始模型使用 Transformer encoder-decoder 结构：

```text
中文 token -> Encoder -> Decoder -> 英文 token
```

实验性的 GPT 路径将翻译改写成一个序列上的 causal language modeling：

```text
[BOS, 中文源文本 token, 英文目标文本 token, EOS]
```

训练时只对英文目标区域计算 loss。推理时模型接收中文前缀，然后自回归生成英文 token。

这样可以在相近的数据和任务上，对比原始 Transformer 与 decoder-only LLM 风格的架构设计。

## 现代组件

decoder-only 模型支持以下可以独立控制的组件：

| 组件 | 配置项 | 现代配置 | 经典对照 |
|---|---|---:|---:|
| Rotary Position Embedding | `use_rope` | RoPE | 正弦位置编码 |
| Grouped-Query Attention | `use_gqa`, `n_kv_head` | 多组 Q 共享 K/V head | 标准多头注意力 |
| Attention Sinks | `use_attention_sink`, `attention_sink_size` | 可学习、始终可见的 K/V 槽位 | 关闭 |
| SwiGLU FFN | `use_swiglu` | SiLU 门控前馈网络 | ReLU FFN |
| RMSNorm | `use_rms_norm` | RMS normalization | LayerNorm |
| Pre-Norm | `use_pre_norm` | 子层计算前归一化 | Post-Norm |

这些选项主要用于教学和消融实验。实现参考了现代 LLM 的常见设计，但不是对 Qwen、LLaMA 或其他生产模型的精确复现。

## 模型架构

### Encoder-Decoder 基线

基线路径包含：

- 正弦位置编码；
- multi-head self-attention；
- encoder-decoder cross-attention；
- ReLU position-wise feed-forward network；
- residual connection 与 LayerNorm；
- greedy 自回归翻译。

仓库通过 Git LFS 跟踪一个预训练 checkpoint：

```text
models/c2e_transformer_[0526-test1].pt
```

### Decoder-Only 实验模型

实验路径包含：

- 中文字符与英文单词共用词表；
- 支持 padding mask 的 causal self-attention；
- 支持 position IDs 的左填充批量生成；
- greedy、beam search、top-k sampling 和 top-p sampling；
- 可配置的 RoPE、GQA、attention sinks、SwiGLU、RMSNorm 和归一化顺序；
- BLEU 评估与 Weights & Biases 日志。

## 项目内容

```text
transformer.py             Encoder-decoder 与可配置 decoder-only 模型
wrap_data.py               Encoder-decoder 数据处理
trainer.py                 Encoder-decoder 训练循环
translator.py              Encoder-decoder 翻译逻辑
train_model.py             Encoder-decoder 训练入口
make_inference.py          使用预训练基线模型进行交互式推理

wrap_data_gpt.py           Decoder-only 序列与词表构造
trainer_gpt.py             Decoder-only 训练和 BLEU 评估
translator_gpt.py          Greedy、beam search 和 sampling 生成
train_gpt.py               Decoder-only 训练入口
c2e_gpt_configs.yaml       现代组件开关与 GPT 训练配置
test_gpt_components.py     组件及经典/现代配置测试

cn-eng.txt                 约 90,000 组中英文句对
input_lang.pkl             基线模型源语言词表
output_lang.pkl            基线模型目标语言词表
models/                    预训练基线和本地训练 checkpoint
```

## 1. 创建运行环境

创建 Conda 环境并安装依赖：

```bash
conda create -n transformer-c2e python=3.8 -y
conda activate transformer-c2e
pip install -r requirements.txt
```

`requirements.txt` 保留了项目原有的 PyTorch 版本。如果该 wheel 与你的 CUDA 或 Python 环境不兼容，请先安装适配本机的 PyTorch，再安装其余依赖。

预训练基线使用 Git LFS。克隆仓库后可以确认 checkpoint 已正确下载：

```bash
git lfs install
git lfs pull
ls -lh models/c2e_transformer_[0526-test1].pt
```

## 2. 运行预训练 Encoder-Decoder 推理

默认命令会使用仓库内置的 checkpoint 和词表：

```bash
python make_inference.py
```

在终端输入中文句子并按回车，即可生成英文翻译。

也可以覆盖模型路径、词表路径和设备：

```bash
python make_inference.py \
  --model_path './models/c2e_transformer_[0526-test1].pt' \
  --input_lang_path './input_lang.pkl' \
  --output_lang_path './output_lang.pkl' \
  --device auto
```

设备支持 `auto`、`cpu` 和 `cuda`。

## 3. 训练 Encoder-Decoder 基线

在 `c2e_configs.yaml` 中填写自己的 W&B entity 和实验参数，然后运行：

```bash
python train_model.py --config_file_path ./c2e_configs.yaml
```

Linux 后台运行示例：

```bash
nohup python -u train_model.py --config_file_path ./c2e_configs.yaml > console.log 2>&1 &
tail -f console.log
```

最终 checkpoint 保存在 `models/`。训练过程中 loss 最优的 state dictionary 保存在 `models/intermediate/`。

## 4. 训练 Decoder-Only 模型

decoder-only 路径使用独立配置：

```bash
python train_gpt.py --config_file_path ./c2e_gpt_configs.yaml
```

训练入口会创建共用词表、训练 causal model、计算验证集 BLEU、在训练结束后评估测试集，并使用多种解码策略打印翻译样例。

新生成的 checkpoint、日志、共用词表以及本地 W&B 运行数据都已被 Git 忽略。

## 5. 组件开关教程

开关位于 `c2e_gpt_configs.yaml` 的 `Model architecture` 部分。

### 现代 LLM 风格配置

```yaml
use_rope: True
use_gqa: True
n_kv_head: 2
use_attention_sink: True
attention_sink_size: 4
use_swiglu: True
use_rms_norm: True
use_pre_norm: True
```

当 `n_head: 8`、`n_kv_head: 2` 时，8 个 query head 共用 2 个 key/value head。`n_head` 必须能够被 `n_kv_head` 整除。

### 经典 Decoder-Only Transformer

```yaml
use_rope: False
use_gqa: False
use_attention_sink: False
use_swiglu: False
use_rms_norm: False
use_pre_norm: False
```

该配置对应：

```text
正弦位置编码 + MHA + ReLU FFN + LayerNorm + Post-Norm
```

当 `use_gqa: False` 时，`n_kv_head` 会被忽略，K/V head 数自动等于 `n_head`。当 `use_attention_sink: False` 时，`attention_sink_size` 会被忽略。

### 单组件消融实验

测量单个组件时，建议从经典配置开始，只打开目标组件。例如只测试 RoPE：

```yaml
use_rope: True
use_gqa: False
use_attention_sink: False
use_swiglu: False
use_rms_norm: False
use_pre_norm: False
```

比较实验时，应保持 random seed、数据划分、模型维度、优化器、学习率、batch size 和 epoch 数不变。

### 推荐对比矩阵

```text
classic
+ RoPE
+ GQA
+ attention sinks
+ SwiGLU
+ RMSNorm
+ Pre-Norm
全部现代组件
```

每次运行应设置不同的 `trial_id`、`ckpt_file_name` 和 `log_file_name`。

## 6. 运行测试

组件测试覆盖：

- SwiGLU 输出形状与反向传播；
- GQA 与可学习 attention sinks；
- 左填充 GPT 前向和反向传播；
- 完全经典的 decoder-only 配置。

运行：

```bash
python -m unittest -v test_gpt_components.py
```

## 7. 数据格式

`cn-eng.txt` 每行包含一组使用 tab 分隔的句对：

```text
中文句子<TAB>英文句子
```

基线模型分别创建源语言和目标语言词表。decoder-only 路径创建一个共用词表，其中包含中文字符、英文单词以及四个特殊 token：

```text
<pad> <bos> <eos> <unk>
```

## 说明

- 仓库内置的预训练 checkpoint 属于原始 encoder-decoder 模型。
- 新训练产生的 checkpoint 默认不会提交到 Git。
- Muon 优化器实验暂不包含在本次发布中。
- 当前生成实现每一步都会重新计算完整序列，尚未使用 KV cache。
