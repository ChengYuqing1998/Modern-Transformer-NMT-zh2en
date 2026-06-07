[English](README.md) | [中文](README.zh-CN.md)

# Modern-Transformer-NMT-zh2en

`Modern-Transformer-NMT-zh2en` 是一个纯 PyTorch 的教育型机器翻译项目，最初基于经典的 encoder-decoder Transformer 实现。

目前仓库包含两条相关的模型路径：

- 经典 encoder-decoder Transformer；
- 用于实验的 decoder-only 翻译模型，可以通过配置开关比较经典 Transformer 组件与现代 LLM 组件。

项目使用 `data/zh-en.txt` 中约 90,000 组中英文句对。注意力、mask、训练目标、生成循环和架构消融都直接实现在仓库中，便于阅读、修改和训练。

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
| Bias | `use_bias` | 保留线性层与 LayerNorm bias | 删除可配置的 bias 参数 |
| Weight tying | `use_weight_tying` | 共享 token embedding 与输出头权重 | 使用两套独立权重 |

这些选项主要用于教学和消融实验。实现参考了现代 LLM 的常见设计，但不是对 Qwen、LLaMA 或其他生产模型的精确复现。

## 模型架构

### Encoder-Decoder 基线

基线路径包含：

- 正弦位置编码；
- multi-head self-attention；
- encoder-decoder cross-attention；
- ReLU position-wise feed-forward network；
- residual connection 与 LayerNorm；
- beam search 自回归翻译。

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
models/transformer.py      Transformer 与 GPT 的全部模型组件
trainer/trainer.py         Encoder-decoder 与 decoder-only trainer
trainer/checkpoint.py      Safetensors 权重与断点续训状态
inference/translator.py    Beam search、greedy 与 sampling
tokenizer/                 标准化、词表、encode/decode 与构建命令
datasets/                  语料读取、tensor 构造与 DataLoader
scripts/                   两个训练入口和统一推理入口
configs/                   按架构分类的 YAML 配置
checkpoints/               按架构分类的 checkpoint
tests/                     两种模型与 tokenizer 测试
data/                      Tab 分隔的中英文语料
```

## Quick Start

创建并激活环境，然后安装依赖：

```bash
conda create -n transformer-nmt python=3.10 -y
conda activate transformer-nmt
pip install -r requirements.txt
```

直接运行两个已经训练好的示例。以下命令默认读取 checkpoint
`metadata.json` 中保存的模型结构、tokenizer 路径和推理参数：

```bash
# Encoder-decoder
python -m scripts.inference \
  --checkpoint checkpoints/encoder_decoder/transformer-example-ckpt \
  --sentence "今天天气很好"

# Decoder-only
python -m scripts.inference \
  --checkpoint checkpoints/decoder_only/gpt-example-ckpt \
  --sentence "今天天气很好"
```

直接训练两种架构：

```bash
python -m scripts.train_encoder_decoder \
  --config-file-path configs/encoder_decoder/c2e_transformer.yaml

python -m scripts.train_decoder_only \
  --config-file-path configs/decoder_only/c2e_gpt.yaml
```

## 1. 创建运行环境

创建 Conda 环境并安装依赖：

```bash
conda create -n transformer-nmt python=3.10 -y
conda activate transformer-nmt
pip install -r requirements.txt
```

请安装与本机 Python、CUDA 和硬件匹配的 PyTorch。CUDA 上使用 BF16
混合精度训练要求 GPU 支持 BF16。CPU 是否支持并能高效执行 BF16 取决于
处理器指令集和 PyTorch 后端；不确定时使用 `fp32_full`。

## 2. 构建 Tokenizer

一次构建 encoder-decoder 的两个词表以及 decoder-only 的统一词表：

```bash
python -m tokenizer.build_tokenizer --architecture all
```

产物位于 `tokenizer/artifacts/`。训练入口也会重新构建并写入对应架构使用的
tokenizer artifact。

## 3. 训练 Encoder-Decoder 基线

```bash
python -m scripts.train_encoder_decoder \
  --config-file-path ./configs/encoder_decoder/c2e_transformer.yaml
```

Linux 后台运行示例：

```bash
nohup python -u -m scripts.train_encoder_decoder --config-file-path ./configs/encoder_decoder/c2e_transformer.yaml > logs/console/transformer.log 2>&1 &
tail -f logs/console/transformer.log
```

Checkpoint 保存在 `checkpoints/encoder_decoder/<trial_name>/`。

## 4. 训练 Decoder-Only 模型

decoder-only 路径使用独立配置：

```bash
python -m scripts.train_decoder_only \
  --config-file-path ./configs/decoder_only/c2e_gpt.yaml
```

训练入口会创建共用词表、训练 causal model 并计算 BLEU。Checkpoint
保存在 `checkpoints/decoder_only/<trial_name>/`。

### Weights & Biases

W&B 是可选功能。仓库中的配置默认使用：

```yaml
wandb_mode: online         # disabled、offline 或 online
wandb_project: modern-transformer-zh-en
wandb_watch_model: True
```

- `disabled`：完全关闭 W&B，不需要账户。
- `offline`：只在已忽略的本地 `wandb/` 目录保存记录，不上传。
- `online`：使用本机保存的凭据上传实验。

不要在 YAML 中填写 API key 或个人 entity。需要在线记录时，只需在本机执行
一次 `wandb login`，凭据会保存在仓库之外。entity 通常可以省略；需要指定
个人或团队时，在本机 shell 中设置：

```bash
export WANDB_ENTITY='your-user-or-team'
export WANDB_MODE='online'
```

`WANDB_MODE`、`WANDB_PROJECT` 和 `WANDB_ENTITY` 会覆盖 YAML 配置。
`.env` 文件已被忽略，`.env.example` 用于说明可用变量。不增加额外依赖时，
可以这样加载本地 `.env`：

```bash
set -a
source .env
set +a
```

## 5. 运行推理

两种翻译架构使用同一个入口。架构、模型配置、序列长度和 tokenizer
路径都从 checkpoint metadata 中读取：

每种架构的 YAML 配置中都包含默认推理参数。训练保存 checkpoint 时，这些
配置会写入 `metadata.json`；独立推理默认读取保存后的值。CLI 参数只覆盖
当前一次调用，优先级为：

```text
CLI 参数 > checkpoint metadata > 代码默认值
```

### 使用 Checkpoint Metadata

这是推荐的默认方式。解码策略、beam size、生成长度、采样参数和 KV cache
开关都从 `metadata.json` 读取：

```bash
# Decoder-only checkpoint
python -m scripts.inference \
  --checkpoint checkpoints/decoder_only/gpt-example-ckpt \
  --sentence "今天天气很好"

# Encoder-decoder checkpoint
python -m scripts.inference \
  --checkpoint checkpoints/encoder_decoder/transformer-example-ckpt \
  --sentence "今天天气很好"
```

### 使用 CLI 覆盖参数

CLI 参数只覆盖当前一次推理中的 checkpoint metadata：

```bash
python -m scripts.inference \
  --checkpoint checkpoints/decoder_only/gpt-example-ckpt \
  --sentence "今天天气很好" \
  --decoding-strategy nucleus_sampling \
  --inference-max-new-tokens 160 \
  --beam-size 5 \
  --temperature 0.8 \
  --top-p 0.9 \
  --top-k 0 \
  --repetition-penalty 1.0 \
  --use-kv-cache
```

### 解码策略

仓库提供三个具名策略：

```yaml
inference_decoding_strategy: beam_search  # greedy、beam_search、nucleus_sampling
beam_size: 5

# nucleus_sampling 使用以下参数。
inference_temperature: 0.8
inference_top_p: 0.9
inference_top_k: 0
inference_repetition_penalty: 1.0
```

- `beam_search` 是翻译和 BLEU 评估的默认策略。对于机器翻译这类由输入约束的
  序列生成任务，它仍然是业界和研究中的成熟选择。
- `nucleus_sampling` 是现代开放式 LLM 的主流采样路径：temperature 调整、
  可选 top-k、top-p nucleus 裁剪、softmax 重新归一化以及 multinomial
  sampling。两种架构均支持，更适合生成多样性，不适合做确定性 BLEU 对比。
- `greedy` 始终选择 logit 最大的 token，是速度最快的确定性基线。

Encoder-decoder 和 decoder-only 都支持以上三种策略、batch 推理和可选
KV cache。需要对照或排错时，可传入 `--no-kv-cache` 使用完整前缀重算。

### PyTorch Deterministic 注意事项

两份训练脚本都会设置 Python、NumPy、PyTorch 和 DataLoader worker 的随机
种子。为了训练性能，严格 deterministic 默认关闭。两个训练入口中保留了
以下注释代码，需要更严格复现时可以取消注释：

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
```

`warn_only=True` 表示遇到不支持确定性实现的算子时只警告，不会中断训练。
不同 GPU、CUDA/PyTorch 版本或 kernel 之间仍不保证逐 bit 完全一致；确定性
算法也可能降低性能。`nucleus_sampling` 本身包含随机采样，需要确定性翻译
对比时应使用 `greedy` 或 `beam_search`。

每个新 checkpoint 都是一个目录。模型权重统一保存在
`model.safetensors`；最佳 checkpoint 仅通过目录名的 `-best` 后缀标识；
可读的模型与运行信息保存在 `metadata.json`；断点续训状态拆分保存在
`trainer_state.pt`、`optimizer.pt`，以及启用 scheduler 时的
`scheduler.pt`。

可以从 epoch 或 step checkpoint 断点续训：

```yaml
resume_from_checkpoint: './checkpoints/decoder_only/c2e-gpt/c2e-gpt-epoch-0002-step-00001200'
```

恢复内容包括模型、optimizer、scheduler、最佳 eval loss、epoch、
optimizer step、epoch 内 micro-batch 位置，以及 Python、NumPy、PyTorch
随机数状态。每个 epoch 的 shuffle 使用确定性种子，因此 step checkpoint
可以从保存时所在 epoch 内继续。

### 训练调度与 Checkpoint 保留

两份训练配置共用以下控制项：

```yaml
gradient_accumulation_steps: 1
train_precision: bf16_mixed       # fp32_full 或 bf16_mixed
checkpoint_precision: bf16        # fp32 或 bf16
print_every_n_steps: 300

eval_strategy: epoch       # epoch 或 step
eval_interval: 1
save_strategy: epoch       # epoch 或 step
save_interval: 1

save_total_limit: 3
save_best: True

show_eval_sample: True
eval_sample_sentence: '今天天气很不错，我早餐吃了一个鸡蛋和一杯牛奶。'
inference_max_new_tokens: 160
inference_use_kv_cache: True
```

两种架构的训练期 BLEU、样例翻译和独立推理统一读取
`inference_max_new_tokens`。命令行参数 `--inference-max-new-tokens` 只覆盖当前一次
推理。

`global_step` 明确定义为一次 optimizer 参数更新。启用梯度累积时，多个
DataLoader batch 是同一个 step 内的 micro-batch。打印固定按 step 触发；
评估和保存可以分别选择按 epoch 或 optimizer step 触发。

所有 checkpoint 目录名都包含 epoch 和 step，例如普通目录
`c2e-gpt-epoch-0002-step-00001200`，最佳目录
`c2e-gpt-epoch-0002-step-00001200-best`。产生新最佳节点时，旧最佳目录会
去掉 `-best` 后缀并成为普通 checkpoint。`save_total_limit` 严格限制目录
总数；当上限为 3 时，如果最近三个节点包含最佳节点，就保留最近三个；
否则保留最佳节点和最近两个普通节点。

`train_precision: bf16_mixed` 使用 PyTorch BF16 autocast，模型主参数和
optimizer state 仍保持 FP32；`fp32_full` 会关闭 autocast。BF16 不需要
loss scaler。

`checkpoint_precision` 与训练策略相互独立。默认设为 `bf16` 时，
`model.safetensors` 中的浮点 tensor 会在写入前转换为 BF16；
`trainer_state.pt`、`optimizer.pt` 和 `scheduler.pt` 会保留断点续训所需的
dtype。如果更重视保存 FP32 模型权重而不是文件大小，可以设置为 `fp32`。

两份默认配置使用相同的模型宽度和深度：

- `configs/encoder_decoder/c2e_transformer.yaml`：51,427,166
  个参数，仅按 BF16 模型权重计算为 98.09 MiB / 102.85 MB。
- `configs/decoder_only/c2e_gpt.yaml`：移除 encoder 后
  保持相同宽度和深度的经典 decoder-only，32,588,320 个参数，BF16 权重为
  62.16 MiB / 65.18 MB。

### RTX 4090 显存参考

使用默认 decoder-only 配置和 `batch_size: 96` 时，RTX 4090 上的实测训练
显存约为 13-16 GB，evaluation 峰值约为 17 GB。这是一次实际运行的参考值，
并非固定上限；序列长度、beam size、KV cache、CUDA/PyTorch 版本和显存分配
器状态都会影响峰值。

如果显存不足，优先降低 `batch_size`。使用 beam search 评估时需要扩展每个
样本的 beam，并保留生成过程和 cache 张量，因此 evaluation 显存可能高于
单个训练 step。

## 6. 组件开关教程

decoder-only 组件开关位于 `configs/decoder_only/c2e_gpt.yaml`。
`use_bias` 和 `use_weight_tying` 则同时存在于两种架构的配置中。

### 现代 LLM 风格配置

```yaml
use_rope: True
use_gqa: True  # 仅打开开关不够，还必须满足 n_kv_head < n_head。
n_kv_head: 2   # 必须整除 n_head；n_head=8 时：1=MQA，2/4=GQA，8=MHA。
use_attention_sink: True
attention_sink_size: 4
use_swiglu: True
use_rms_norm: True
use_pre_norm: True
use_bias: False
use_weight_tying: False
```

不能只设置 `use_gqa: True`：`n_kv_head` 必须小于 `n_head`，并且
`n_head` 必须能够被 `n_kv_head` 整除。当 `n_head: 8`、
`n_kv_head: 2` 时，8 个 query head 共用 2 个 key/value head。
`n_kv_head: 8` 仍然属于普通 MHA，因此模型构建器会拒绝它与
`use_gqa: True` 同时使用。

### 经典 Decoder-Only Transformer

```yaml
use_rope: False
use_gqa: False
use_attention_sink: False
use_swiglu: False
use_rms_norm: False
use_pre_norm: False
use_bias: False
use_weight_tying: False
```

该配置对应：

```text
正弦位置编码 + MHA + ReLU FFN + LayerNorm + Post-Norm
```

当 `use_gqa: False` 时，`n_kv_head` 会被忽略，K/V head 数自动等于 `n_head`。当 `use_attention_sink: False` 时，`attention_sink_size` 会被忽略。

`use_bias: False` 会关闭 attention projection、FFN 线性层、输出头以及可配置
LayerNorm beta 的 bias 参数。`use_weight_tying: True` 会共享 decoder token
embedding 与输出 projection；encoder-decoder 只共享 decoder embedding 和
目标输出头，decoder-only 则共享统一词表 embedding 和 LM head。

这两个开关会改变模型参数结构，推理时必须与训练时保持一致。新训练保存的
checkpoint metadata 会记录它们；加载已有 checkpoint 时擅自修改开关可能
导致输出异常或 state dict 无法加载。

### 单组件消融实验

测量单个组件时，建议从经典配置开始，只打开目标组件。例如只测试 RoPE：

```yaml
use_rope: True
use_gqa: False
use_attention_sink: False
use_swiglu: False
use_rms_norm: False
use_pre_norm: False
use_bias: False
use_weight_tying: False
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

每次运行只需要设置不同的 `trial_name`。它同时控制 W&B run 名称、
checkpoint 子目录、checkpoint 前缀和日志文件名。

## 7. 运行测试

组件测试覆盖：

- SwiGLU 输出形状与反向传播；
- GQA 与可学习 attention sinks；
- 左填充 GPT 前向和反向传播；
- 完全经典的 decoder-only 配置。

运行：

```bash
python -m unittest discover -v
```

## 8. 数据格式

`data/zh-en.txt` 每行包含一组使用 tab 分隔的句对：

```text
中文句子<TAB>英文句子
```

基线模型分别创建源语言和目标语言词表。decoder-only 路径创建一个共用词表，其中包含中文字符、英文单词以及四个特殊 token：

```text
<pad> <bos> <eos> <unk>
```

数据长度参数会直接体现它所测量的对象：

```yaml
# 与 len(target_sentence.split()) 比较，当前使用小于等于。
max_target_sentence_split_length: 128

# 使用 tokenizer token ID 计数，表示 padding 后序列长度的下限。
# 如果样本需要更多 token，该长度会自动增大。
min_sequence_token_length: 32

# Decoder-only 的硬性上下文上限，使用 tokenizer token ID 计数。
# BOS + 源文本 token + 目标文本 token + EOS 必须不超过该值。
max_context_len: 512
```

## 9. 推理与 KV Cache

两种架构及正式推理路径现在都支持可选 KV cache。

- Decoder-only 的 greedy、sampling 和 beam search 会在 prompt prefill 后
  缓存每层 causal self-attention K/V，后续每步只处理最新 token。
- Encoder-decoder beam search 只计算一次 encoder，并缓存 decoder
  self-attention K/V 以及每层固定的 cross-attention K/V projection。
- 两种 beam search 每次选择或替换 beam 时都会同步重排所有层的 cache。
- Decoder-only 中的 GQA、RoPE、左填充和 attention sinks 需要配套处理
  cache position ID 与 mask，当前实现已经覆盖这些情况。
- 模型默认 `use_cache=False`，因此训练路径不受影响。

设置 `inference_use_kv_cache: False` 或传入 `--no-kv-cache`，可以与完整前缀
重算进行对照。

## 说明

- 训练生成的 checkpoint 目录应保留在本地；只有明确发布某个模型时才应将其
  加入版本控制。
- Muon 优化器实验仍然是可选功能，不属于默认 decoder-only 配置。
