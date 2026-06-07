[English](README.md) | [中文](README.zh-CN.md)

# Modern-Transformer-NMT-zh2en

`Modern-Transformer-NMT-zh2en` is a pure-PyTorch educational machine translation project built around the original encoder-decoder Transformer.

The repository contains two related model paths:

- a classic encoder-decoder Transformer;
- a decoder-only translation model for comparing classic Transformer blocks with modern LLM components through configuration switches.

The project uses approximately 90,000 Chinese-English sentence pairs from `data/zh-en.txt`. The implementation is intentionally kept local and readable so the attention mechanism, masks, training targets, generation loop, and architectural ablations can be inspected directly.

## Why Two Architectures?

The original model follows the Transformer encoder-decoder design:

```text
Chinese tokens -> Encoder -> Decoder -> English tokens
```

The experimental GPT path reformulates translation as causal language modeling over one sequence:

```text
[BOS, Chinese source tokens, English target tokens, EOS]
```

During training, loss is applied only to the English target region. During inference, the model receives the Chinese prefix and autoregressively generates English tokens.

This makes the repository useful for comparing the original Transformer with decoder-only LLM-style design choices while keeping the dataset and task similar.

## Modern Components

The decoder-only model supports the following independently configurable components:

| Component | Configuration | Modern setting | Classic comparison |
|---|---|---:|---:|
| Rotary Position Embedding | `use_rope` | RoPE | sinusoidal position encoding |
| Grouped-Query Attention | `use_gqa`, `n_kv_head` | shared K/V heads | standard multi-head attention |
| Attention Sinks | `use_attention_sink`, `attention_sink_size` | learned always-visible K/V slots | disabled |
| SwiGLU FFN | `use_swiglu` | SiLU-gated feed-forward network | ReLU FFN |
| RMSNorm | `use_rms_norm` | RMS normalization | LayerNorm |
| Pre-Norm | `use_pre_norm` | normalize before each sublayer | Post-Norm |
| Bias | `use_bias` | keep linear and LayerNorm bias terms | remove configurable bias terms |
| Weight tying | `use_weight_tying` | share token embedding and output-head weights | use separate weights |

These options are intended for educational ablation experiments. The implementation is inspired by common modern LLM designs, but it is not an exact reproduction of Qwen, LLaMA, or any other production model.

## Architecture

### Encoder-Decoder Baseline

The baseline path contains:

- sinusoidal positional encoding;
- multi-head self-attention;
- encoder-decoder cross-attention;
- ReLU position-wise feed-forward networks;
- residual connections and LayerNorm;
- beam-search autoregressive translation.

### Decoder-Only Experimental Model

The experimental path contains:

- a unified Chinese-character and English-word vocabulary;
- causal self-attention with padding-mask support;
- left-padded batch generation with position IDs;
- greedy decoding, beam search, top-k sampling, and top-p sampling;
- configurable RoPE, GQA, attention sinks, SwiGLU, RMSNorm, and normalization order;
- BLEU evaluation and Weights & Biases logging.

## Contents

```text
models/transformer.py      All Transformer and GPT model components
trainer/trainer.py         Encoder-decoder and decoder-only trainers
trainer/checkpoint.py      Safetensors weights and resumable trainer state
inference/translator.py    Beam search, greedy decoding, and sampling
tokenizer/                 Normalization, vocabulary, encode/decode, and builds
datasets/                  Corpus loading, tensor construction, and DataLoaders
scripts/                   Two training commands and unified inference
configs/                   Per-architecture YAML configurations
checkpoints/               Per-architecture model checkpoints
tests/                     Encoder-decoder, decoder-only, and tokenizer tests
data/                      Tab-separated Chinese-English corpus
```

## Quick Start

Create and activate the environment, then install the dependencies:

```bash
conda create -n transformer-nmt python=3.10 -y
conda activate transformer-nmt
pip install -r requirements.txt
```

Run the two pretrained examples. Both commands use the model architecture,
tokenizer paths, and inference defaults stored in checkpoint
`metadata.json`:

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

Train either architecture directly:

```bash
python -m scripts.train_encoder_decoder \
  --config-file-path configs/encoder_decoder/c2e_transformer.yaml

python -m scripts.train_decoder_only \
  --config-file-path configs/decoder_only/c2e_gpt.yaml
```

## 1. Create The Environment

Create a Conda environment and install the dependencies:

```bash
conda create -n transformer-nmt python=3.10 -y
conda activate transformer-nmt
pip install -r requirements.txt
```

Install a PyTorch build compatible with the local Python, CUDA, and hardware.
BF16 mixed-precision CUDA training requires a BF16-capable GPU. CPU BF16
availability and performance depend on the processor and PyTorch backend;
`fp32_full` is the portable fallback.

## 2. Build Tokenizers

Build the separate encoder-decoder vocabularies and the unified decoder-only
vocabulary:

```bash
python -m tokenizer.build_tokenizer --architecture all
```

The files are written to `tokenizer/artifacts/`. Training also rebuilds and
writes the tokenizer artifacts used by that architecture.

## 3. Train The Encoder-Decoder Baseline

```bash
python -m scripts.train_encoder_decoder \
  --config-file-path ./configs/encoder_decoder/c2e_transformer.yaml
```

For a background Linux process:

```bash
nohup python -u -m scripts.train_encoder_decoder --config-file-path ./configs/encoder_decoder/c2e_transformer.yaml > logs/console/transformer.log 2>&1 &
tail -f logs/console/transformer.log
```

Checkpoints are written under
`checkpoints/encoder_decoder/<trial_name>/`.

## 4. Train The Decoder-Only Model

The decoder-only path uses its own configuration:

```bash
python -m scripts.train_decoder_only \
  --config-file-path ./configs/decoder_only/c2e_gpt.yaml
```

It builds the unified vocabulary, trains the causal model, and evaluates BLEU.
Checkpoints are written under `checkpoints/decoder_only/<trial_name>/`.

### Weights & Biases

W&B is optional. Tracked configs default to:

```yaml
wandb_mode: online         # disabled, offline, or online
wandb_project: modern-transformer-zh-en
wandb_watch_model: True
```

- `disabled`: no W&B logging and no account is required.
- `offline`: records runs under the ignored local `wandb/` directory without
  uploading.
- `online`: uploads runs using credentials stored on your machine.

Do not add an API key or personal entity to YAML. For online logging, run
`wandb login` once; the credential is stored outside this repository. Entity
is optional. If a team or explicit account is required, set it only in your
shell:

```bash
export WANDB_ENTITY='your-user-or-team'
export WANDB_MODE='online'
```

`WANDB_MODE`, `WANDB_PROJECT`, and `WANDB_ENTITY` override YAML settings.
`.env` files are ignored; `.env.example` documents the available variables.
To use a local `.env` without adding a dependency:

```bash
set -a
source .env
set +a
```

## 5. Run Inference

Both translation architectures use the same command. The architecture, model
configuration, sequence length, and tokenizer paths are read from checkpoint
metadata:

Inference defaults are defined in each architecture's YAML configuration and
are copied into checkpoint `metadata.json` during training. Standalone
inference reads those saved values by default. CLI arguments override them
only for the current invocation:

```text
CLI argument > checkpoint metadata > code default
```

### Use Checkpoint Metadata

This is the recommended default. Decoding strategy, beam size, generation
length, sampling parameters, and KV-cache behavior come from `metadata.json`:

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

### Override From The CLI

CLI arguments override checkpoint metadata for one invocation:

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

### Decoding Strategies

The repository exposes three named strategies:

```yaml
inference_decoding_strategy: beam_search  # greedy, beam_search, nucleus_sampling
beam_size: 5

# Used by nucleus_sampling.
inference_temperature: 0.8
inference_top_p: 0.9
inference_top_k: 0
inference_repetition_penalty: 1.0
```

- `beam_search` is the default for translation and BLEU evaluation. It remains
  a standard choice for input-grounded sequence generation such as machine
  translation.
- `nucleus_sampling` is available for both architectures and applies
  temperature scaling, optional top-k filtering, top-p nucleus filtering,
  softmax renormalization, and multinomial sampling. It is useful for diverse
  translation candidates rather than deterministic BLEU comparison.
- `greedy` always selects the highest-logit token and is the fastest
  deterministic baseline.

Both encoder-decoder and decoder-only inference support all three strategies,
batch generation, and optional KV cache. Use `--no-kv-cache` for full-prefix
recomputation when comparing behavior or debugging.

### Deterministic Execution

Both training scripts seed Python, NumPy, PyTorch, and DataLoader workers.
Strict deterministic execution is disabled by default for performance. Each
training entry point contains a commented block that can enable:

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
```

`warn_only=True` means unsupported nondeterministic operations emit a warning
instead of stopping training. Exact bitwise reproducibility is not guaranteed
across different GPUs, CUDA/PyTorch versions, or kernels. Deterministic
algorithms can also reduce performance. `nucleus_sampling` remains stochastic;
use `greedy` or `beam_search` for deterministic translation comparisons.

Each new checkpoint is a directory. All model weights are stored in
`model.safetensors`; the `-best` directory suffix marks the best checkpoint.
Readable model and run metadata is stored in
`metadata.json`; resumable training state is split across `trainer_state.pt`,
`optimizer.pt`, and, when enabled, `scheduler.pt`.

Training can resume from either an epoch or step checkpoint:

```yaml
resume_from_checkpoint: './checkpoints/decoder_only/c2e-gpt/c2e-gpt-epoch-0002-step-00001200'
```

Resume restores model, optimizer, scheduler, best evaluation loss, epoch,
optimizer step, micro-batch position, and Python/NumPy/PyTorch RNG states.
Epoch shuffling is seeded deterministically so a step checkpoint can continue
inside the saved epoch.

### Training Schedule And Checkpoint Retention

Both training configs use the same controls:

```yaml
gradient_accumulation_steps: 1
train_precision: bf16_mixed       # fp32_full or bf16_mixed
checkpoint_precision: bf16        # fp32 or bf16
print_every_n_steps: 300

eval_strategy: epoch       # epoch or step
eval_interval: 1
save_strategy: epoch       # epoch or step
save_interval: 1

save_total_limit: 3
save_best: True

show_eval_sample: True
eval_sample_sentence: '今天天气很不错，我早餐吃了一个鸡蛋和一杯牛奶。'
inference_max_new_tokens: 160
inference_use_kv_cache: True
```

`inference_max_new_tokens` is shared by training-time BLEU evaluation, sample
translation, and standalone inference for both architectures. The CLI option
`--inference-max-new-tokens` overrides the checkpoint value for one invocation.

A `global_step` means one optimizer parameter update. With gradient
accumulation enabled, several DataLoader batches are micro-batches inside one
step. Printing is always step-based; evaluation and saving can independently
use epochs or optimizer steps.

Every checkpoint includes both counters in its directory name. A regular
checkpoint is named `c2e-gpt-epoch-0002-step-00001200`; a best checkpoint is
named `c2e-gpt-epoch-0002-step-00001200-best`. When evaluation finds a new
best loss, the previous best directory loses its `-best` suffix and becomes a
regular checkpoint. A regular checkpoint for the new best node is removed to
avoid duplication. `save_total_limit` strictly limits all checkpoint
directories. With a limit of 3, the trainer keeps the three most recent
checkpoints when that set contains the best checkpoint; otherwise it keeps the
best checkpoint and the two most recent regular checkpoints.

`train_precision: bf16_mixed` enables PyTorch BF16 autocast while model master
parameters and optimizer state remain FP32. `fp32_full` disables autocast.
BF16 does not use a loss scaler.

`checkpoint_precision` is independent from the training strategy. With the
default `bf16`, floating-point tensors in `model.safetensors` are converted
to BF16 before writing.
`trainer_state.pt`, `optimizer.pt`, and `scheduler.pt`
retain the dtypes required for resuming. Set it to `fp32` when preserving FP32
model weights is more important than checkpoint size.

The two default configurations use the same width and depth:

- `configs/encoder_decoder/c2e_transformer.yaml`: 51,427,166
  parameters, 98.09 MiB / 102.85 MB when counting BF16 model weights only.
- `configs/decoder_only/c2e_gpt.yaml`: the matching
  decoder-only width/depth after removing the encoder, 32,588,320 parameters,
  62.16 MiB / 65.18 MB as BF16 weights.

### RTX 4090 Memory Reference

With the default decoder-only configuration and `batch_size: 96`, an observed
RTX 4090 run used approximately 13-16 GB during training and peaked around
17 GB during evaluation. These are empirical figures, not guaranteed limits;
sequence lengths, beam size, KV cache, CUDA/PyTorch versions, and allocator
state can change the peak.

If GPU memory is insufficient, reduce `batch_size` first. Evaluation with beam
search can use more memory than a training step because it expands each sample
into multiple beams and retains generation/cache tensors.

## 6. Component Switch Tutorial

The decoder-only component switches are in
`configs/decoder_only/c2e_gpt.yaml`. `use_bias` and `use_weight_tying` are
available in both architecture configurations.

### Modern LLM-Style Configuration

```yaml
use_rope: True
use_gqa: True  # GQA also requires n_kv_head < n_head.
n_kv_head: 2   # Must divide n_head; with n_head=8: 1=MQA, 2/4=GQA, 8=MHA.
use_attention_sink: True
attention_sink_size: 4
use_swiglu: True
use_rms_norm: True
use_pre_norm: True
use_bias: False
use_weight_tying: False
```

Setting only `use_gqa: True` is not sufficient: `n_kv_head` must be smaller
than `n_head`, and `n_head` must be divisible by `n_kv_head`. With
`n_head: 8` and `n_kv_head: 2`, eight query heads share two key/value heads.
Using `n_kv_head: 8` remains ordinary MHA, so the model builder rejects that
combination when `use_gqa: True`.

### Classic Decoder-Only Transformer

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

This selects:

```text
sinusoidal positions + MHA + ReLU FFN + LayerNorm + Post-Norm
```

When `use_gqa: False`, `n_kv_head` is ignored and the number of K/V heads equals `n_head`. When `use_attention_sink: False`, `attention_sink_size` is ignored.

`use_bias: False` removes bias parameters from attention projections,
feed-forward linear layers, output heads, and configurable LayerNorm beta
terms. `use_weight_tying: True` shares the decoder token embedding with the
output projection; in encoder-decoder models only the decoder embedding and
target output head are tied. Decoder-only models tie their unified token
embedding to the language-model head.

These switches change the model parameter structure. Inference must use the
same values that were used for training. Checkpoint metadata preserves them
for newly trained models; changing either switch while loading an existing
checkpoint can produce invalid output or a state-dict mismatch.

### One-Component Ablation

To measure one component, start from the classic configuration and enable only that switch. For example, a RoPE-only experiment is:

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

Keep the random seed, data split, model dimensions, optimizer, learning rate, batch size, and epoch count unchanged when comparing experiments.

### Suggested Comparison Matrix

```text
classic
+ RoPE
+ GQA
+ attention sinks
+ SwiGLU
+ RMSNorm
+ Pre-Norm
all modern components
```

Use a different `trial_name` for each run. It controls the W&B run name,
checkpoint subdirectory, checkpoint prefix, and log filename.

## 7. Run Tests

The component tests cover:

- SwiGLU shape and backward propagation;
- GQA with learned attention sinks;
- left-padded GPT forward/backward behavior;
- the fully classic decoder-only configuration.

Run:

```bash
python -m unittest discover -v
```

## 8. Data Format

`data/zh-en.txt` contains one tab-separated sentence pair per line:

```text
Chinese sentence<TAB>English sentence
```

The baseline uses separate source and target vocabularies. The decoder-only path creates one unified vocabulary containing Chinese characters, English words, and four special tokens:

```text
<pad> <bos> <eos> <unk>
```

Dataset length controls are named after what they actually measure:

```yaml
# Compared with len(target_sentence.split()); the comparison is <=.
max_target_sentence_split_length: 128

# Minimum padded sequence length measured in tokenizer token IDs.
# The value grows automatically when a sample needs more tokens.
min_sequence_token_length: 32

# Decoder-only hard context limit in tokenizer token IDs.
# BOS + source tokens + target tokens + EOS must fit within this value.
max_context_len: 512
```

## 9. Inference And KV Cache

Both architectures and their production inference paths support optional KV
cache decoding.

- Decoder-only greedy, sampling, and beam search cache each layer's causal
  self-attention K/V and process only the latest token after prompt prefill.
- Encoder-decoder beam search computes the encoder once, then caches decoder
  self-attention K/V and each layer's fixed cross-attention K/V projections.
- Both beam-search implementations reorder every layer's cached tensors when
  beams are selected or replaced.
- GQA, RoPE, left padding, and attention sinks require cache-aware position
  IDs and mask handling; these are handled by the decoder-only cache path.
- Training remains unchanged because `use_cache=False` is the model default.

Set `inference_use_kv_cache: False` or pass `--no-kv-cache` to compare against
full-prefix recomputation.

## Notes

- Generated checkpoint directories should be kept local unless a specific
  model release is intentionally added to version control.
- Muon optimizer experiments remain optional and are not part of the default decoder-only configuration.
