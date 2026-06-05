[English](README.md) | [中文](README.zh-CN.md)

# Modern-Transformer-NMT-zh2en

`Modern-Transformer-NMT-zh2en` is a pure-PyTorch educational machine translation project built around the original encoder-decoder Transformer.

The repository now contains two related model paths:

- a classic encoder-decoder Transformer with a pretrained checkpoint for immediate Chinese-to-English inference;
- a decoder-only translation model for comparing classic Transformer blocks with modern LLM components through configuration switches.

The project uses approximately 90,000 Chinese-English sentence pairs from `data/cn-eng.txt`. The implementation is intentionally kept local and readable so the attention mechanism, masks, training targets, generation loop, and architectural ablations can be inspected directly.

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

These options are intended for educational ablation experiments. The implementation is inspired by common modern LLM designs, but it is not an exact reproduction of Qwen, LLaMA, or any other production model.

## Architecture

### Encoder-Decoder Baseline

The baseline path contains:

- sinusoidal positional encoding;
- multi-head self-attention;
- encoder-decoder cross-attention;
- ReLU position-wise feed-forward networks;
- residual connections and LayerNorm;
- greedy autoregressive translation.

A pretrained checkpoint is tracked with Git LFS:

```text
models/c2e_transformer_[0526-test1].pt
```

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
transformer.py             Encoder-decoder and configurable decoder-only models
wrap_data.py               Encoder-decoder data pipeline
trainer.py                 Encoder-decoder training loop
translator.py              Encoder-decoder translation logic
train_model.py             Encoder-decoder training entry point
make_inference.py          Interactive inference with the pretrained baseline

decoder_only/data.py       Decoder-only sequence and vocabulary construction
decoder_only/trainer.py    Decoder-only training and BLEU evaluation
decoder_only/generation.py Greedy, beam-search, and sampling generation
train_gpt.py               Backward-compatible decoder-only training entry point

configs/                   Baseline and decoder-only YAML configurations
data/                      Parallel corpus and serialized vocabularies
tests/                     Component and classic/modern configuration tests
models/                    Pretrained baseline and local checkpoints
logs/                      Generated training and console logs
```

## 1. Create The Environment

Create a Conda environment and install the dependencies:

```bash
conda create -n transformer-c2e python=3.8 -y
conda activate transformer-c2e
pip install -r requirements.txt
```

The repository keeps its historical PyTorch dependency in `requirements.txt`. If that wheel does not match your CUDA or Python installation, install a compatible PyTorch build first and then install the remaining packages.

The pretrained baseline uses Git LFS. After cloning, verify that the checkpoint has been downloaded:

```bash
git lfs install
git lfs pull
ls -lh models/c2e_transformer_[0526-test1].pt
```

## 2. Run Pretrained Encoder-Decoder Inference

The default command uses the included checkpoint and vocabulary files:

```bash
python make_inference.py
```

Enter a Chinese sentence in the terminal and press Enter to generate its English translation.

Paths and device selection can be overridden:

```bash
python make_inference.py \
  --model_path './models/c2e_transformer_[0526-test1].pt' \
  --input_lang_path './data/input_lang.pkl' \
  --output_lang_path './data/output_lang.pkl' \
  --device auto
```

Supported device values are `auto`, `cpu`, and `cuda`.

## 3. Train The Encoder-Decoder Baseline

Set your W&B entity and experiment values in `configs/c2e_transformer.yaml`, then run:

```bash
python train_model.py --config_file_path ./configs/c2e_transformer.yaml
```

For a background Linux process:

```bash
nohup python -u train_model.py --config_file_path ./configs/c2e_transformer.yaml > logs/transformer-console.log 2>&1 &
tail -f logs/transformer-console.log
```

Final checkpoints are written under `models/`. Best intermediate state dictionaries are written under `models/intermediate/`.

## 4. Train The Decoder-Only Model

The decoder-only path uses its own configuration:

```bash
python train_gpt.py --config_file_path ./configs/c2e_gpt.yaml
```

It builds a unified vocabulary, trains the causal model, evaluates validation BLEU, evaluates the test split after training, and prints example translations using multiple decoding strategies.

Generated checkpoints, logs, the generated unified vocabulary, and local W&B runs are ignored by Git.

## 5. Component Switch Tutorial

The switches are under the `Model architecture` section of `configs/c2e_gpt.yaml`.

### Modern LLM-Style Configuration

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

With `n_head: 8` and `n_kv_head: 2`, eight query heads share two key/value heads. `n_head` must be divisible by `n_kv_head`.

### Classic Decoder-Only Transformer

```yaml
use_rope: False
use_gqa: False
use_attention_sink: False
use_swiglu: False
use_rms_norm: False
use_pre_norm: False
```

This selects:

```text
sinusoidal positions + MHA + ReLU FFN + LayerNorm + Post-Norm
```

When `use_gqa: False`, `n_kv_head` is ignored and the number of K/V heads equals `n_head`. When `use_attention_sink: False`, `attention_sink_size` is ignored.

### One-Component Ablation

To measure one component, start from the classic configuration and enable only that switch. For example, a RoPE-only experiment is:

```yaml
use_rope: True
use_gqa: False
use_attention_sink: False
use_swiglu: False
use_rms_norm: False
use_pre_norm: False
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

Use a different `trial_id`, `ckpt_file_name`, and `log_file_name` for each run.

## 6. Run Tests

The component tests cover:

- SwiGLU shape and backward propagation;
- GQA with learned attention sinks;
- left-padded GPT forward/backward behavior;
- the fully classic decoder-only configuration.

Run:

```bash
python -m unittest -v tests/test_gpt_components.py
```

## 7. Data Format

`data/cn-eng.txt` contains one tab-separated sentence pair per line:

```text
Chinese sentence<TAB>English sentence
```

The baseline uses separate source and target vocabularies. The decoder-only path creates one unified vocabulary containing Chinese characters, English words, and four special tokens:

```text
<pad> <bos> <eos> <unk>
```

## Notes

- The included pretrained checkpoint belongs to the original encoder-decoder model.
- Newly trained checkpoints are intentionally not tracked.
- Muon optimizer experiments are intentionally excluded from this release.
- The current generation implementation recomputes the full sequence at each step and does not yet use a KV cache.
