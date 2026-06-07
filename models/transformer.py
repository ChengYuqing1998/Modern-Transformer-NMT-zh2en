import torch.nn as nn
import torch
import math


PAD_token = 0
BOS_token = 1
EOS_token = 2
UNK_token = 3


def _vocab_size(tokenizer_or_size):
    if isinstance(tokenizer_or_size, int):
        return tokenizer_or_size
    return len(tokenizer_or_size.index2word)


def _resolve_device(configs):
    device = configs.get("_device")
    if device is not None:
        return torch.device(device)
    value = configs.get("device", "cpu")
    if isinstance(value, torch.device):
        return value
    if value in ("cpu", "cuda", "mps"):
        return torch.device(value)
    return eval(value, {"torch": torch})


def build_model(
    input_tokenizer,
    output_tokenizer,
    max_src_len,
    max_trg_len,
    configs: dict,
):
    enc_vocab_size = _vocab_size(input_tokenizer)
    dec_vocab_size = _vocab_size(output_tokenizer)
    model = Transformer(enc_voc_size=enc_vocab_size,
                        dec_voc_size=dec_vocab_size,
                        generator=Generator(
                            configs['model_dim'],
                            dec_vocab_size,
                            use_bias=configs.get('use_bias', True),
                        ),
                        model_dim=configs['model_dim'],
                        n_head=configs['n_head'],
                        max_src_len=max_src_len,
                        max_trg_len=max_trg_len,
                        hidden_dim=configs['hidden_dim'],
                        n_layer=configs['n_layer'],
                        drop_prob=configs['drop_prob'],
                        src_pad_idx=configs['pad_token'],
                        trg_pad_idx=configs['pad_token'],
                        src_bos_idx=configs['bos_token'],
                        trg_bos_idx=configs['bos_token'],
                        src_eos_idx=configs['eos_token'],
                        trg_eos_idx=configs['eos_token'],
                        device=_resolve_device(configs),
                        use_bias=configs.get('use_bias', True),
                        use_weight_tying=configs.get('use_weight_tying', True))
    return model


class PositionLayer(nn.Module):
    def __init__(self, max_len, embed_dim, device):
        super(PositionLayer, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.encoding = torch.zeros(max_len, embed_dim, requires_grad=False, device=device)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) /embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].requires_grad_(False)


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, dropout=None, mask=None, e=1e9):  # fixed the eps
        tensor_dim = k.size(-1)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(tensor_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)  # keep the shape of mask since the broadcast

        score = self.softmax(score)
        if dropout is not None:
            score = dropout(score)  # dropout attention

        context = score @ v
        return context, score


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, drop_prob, device, use_bias=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = self.embed_dim // self.n_head

        self.linear_q = nn.Linear(
            self.embed_dim, self.head_dim * self.n_head, bias=use_bias, device=device
        )
        self.linear_k = nn.Linear(
            self.embed_dim, self.head_dim * self.n_head, bias=use_bias, device=device
        )
        self.linear_v = nn.Linear(
            self.embed_dim, self.head_dim * self.n_head, bias=use_bias, device=device
        )
        self.scaled_dot_product_attention = ScaleDotProductAttention()

        self.linear_attention = nn.Linear(
            self.head_dim * self.n_head, self.embed_dim, bias=use_bias, device=device
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        past_key_value=None,
        use_cache=False,
        static_key_value=False,
    ):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # add a dimension into mask
        batch_size = q.size(0)
        query = self.linear_q(q).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        if static_key_value and past_key_value is not None:
            key, value = past_key_value
        else:
            key = self.linear_k(k).view(
                batch_size, -1, self.n_head, self.head_dim
            ).transpose(1, 2)
            value = self.linear_v(v).view(
                batch_size, -1, self.n_head, self.head_dim
            ).transpose(1, 2)
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key = torch.cat((past_key, key), dim=2)
                value = torch.cat((past_value, value), dim=2)
        present_key_value = (key, value) if use_cache else None
        context, _ = self.scaled_dot_product_attention(query, key, value, self.dropout, mask)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.head_dim * self.n_head
        )
        output = self.linear_attention(context)
        if use_cache:
            return output, present_key_value
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, drop_prob, device, use_bias=True):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, hidden_dim, bias=use_bias, device=device)
        self.linear2 = nn.Linear(hidden_dim, model_dim, bias=use_bias, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU activation function used in Qwen and other modern LLMs.
    Based on GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202)

    SwiGLU(x) = Swish(W1 * x) ⊗ (W2 * x)
    where Swish(x) = x * sigmoid(x)
    """
    def __init__(self, model_dim, hidden_dim, drop_prob, device):
        super(SwiGLU, self).__init__()
        # For SwiGLU, we need two separate projections: gate and up
        # Then project back down
        self.gate_proj = nn.Linear(model_dim, hidden_dim, bias=False, device=device)
        self.up_proj = nn.Linear(model_dim, hidden_dim, bias=False, device=device)
        self.down_proj = nn.Linear(hidden_dim, model_dim, bias=False, device=device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # Gate branch with SiLU/Swish activation
        gate = torch.nn.functional.silu(self.gate_proj(x))
        # Up projection branch
        up = self.up_proj(x)
        # Element-wise multiplication (Gated Linear Unit)
        hidden = gate * up
        # Dropout
        hidden = self.dropout(hidden)
        # Down projection
        output = self.down_proj(hidden)
        return output


class LayerNorm(nn.Module):
    def __init__(self, model_dim, device, eps=1e-6, use_bias=True):  # fixed the eps
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim, device=device))
        self.use_bias = use_bias
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(model_dim, device=device))
        else:
            self.register_parameter("beta", None)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out
        if self.beta is not None:
            out = out + self.beta
        return out


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    Used in modern LLMs like Qwen, LLaMA for better efficiency

    RMSNorm is simpler than LayerNorm:
    - No mean centering (no bias term)
    - Only normalizes by RMS (root mean square)
    - Faster and often works just as well
    """
    def __init__(self, hidden_size, device=None, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.model_dim = model_dim
        self.tok_embedding = nn.Embedding(vocab_size, model_dim, device=device)
        self.pos_embedding = PositionLayer(max_len, model_dim, device=device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, position_ids=None):
        tok_embedding = self.tok_embedding(x) * math.sqrt(self.model_dim)  # multiply square root of model dim
        if position_ids is None:
            pos_embedding = self.pos_embedding(x)
        else:
            pos_embedding = self.pos_embedding.pe[0, position_ids]
        return self.dropout(tok_embedding + pos_embedding)


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, n_head, drop_prob, device, use_bias=True):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(
            model_dim, n_head, drop_prob, device=device, use_bias=use_bias
        )
        self.layer_norm1 = LayerNorm(model_dim, device=device, use_bias=use_bias)
        self.dropout1 = nn.Dropout(drop_prob)
        self.feed_forward = PositionWiseFeedForward(
            model_dim, hidden_dim, drop_prob, device=device, use_bias=use_bias
        )
        self.layer_norm2 = LayerNorm(model_dim, device=device, use_bias=use_bias)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, s_mask):
        # the norm order seems to matter
        xnorm = self.layer_norm1(x)
        x = x + self.dropout1(self.attention(xnorm, xnorm, xnorm, mask=s_mask))
        xnorm = self.layer_norm2(x)
        x = x + self.dropout2(self.feed_forward(xnorm))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, n_head, drop_prob, device, use_bias=True):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            model_dim, n_head, drop_prob, device=device, use_bias=use_bias
        )
        self.layer_norm1 = LayerNorm(model_dim, device=device, use_bias=use_bias)
        self.dropout1 = nn.Dropout(drop_prob)
        self.enc_dec_attention = MultiHeadAttention(
            model_dim, n_head, drop_prob, device=device, use_bias=use_bias
        )
        self.layer_norm2 = LayerNorm(model_dim, device=device, use_bias=use_bias)
        self.dropout2 = nn.Dropout(drop_prob)
        self.feed_forward = PositionWiseFeedForward(
            model_dim, hidden_dim, drop_prob, device=device, use_bias=use_bias
        )
        self.layer_norm3 = LayerNorm(model_dim, device=device, use_bias=use_bias)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(
        self,
        dec,
        enc,
        s_mask,
        t_mask,
        past_key_value=None,
        use_cache=False,
    ):
        # the norm order seems to matter
        self_past = None
        cross_past = None
        if past_key_value is not None:
            self_past, cross_past = past_key_value

        decnorm = self.layer_norm1(dec)
        self_attention_output = self.self_attention(
            decnorm,
            decnorm,
            decnorm,
            mask=t_mask,
            past_key_value=self_past,
            use_cache=use_cache,
        )
        if use_cache:
            self_attention_output, present_self = self_attention_output
        dec = dec + self.dropout1(self_attention_output)

        decnorm = self.layer_norm2(dec)
        cross_attention_output = self.enc_dec_attention(
            decnorm,
            enc,
            enc,
            mask=s_mask,
            past_key_value=cross_past,
            use_cache=use_cache,
            static_key_value=True,
        )
        if use_cache:
            cross_attention_output, present_cross = cross_attention_output
        dec_lookup = dec + self.dropout2(cross_attention_output)

        declookup_norm = self.layer_norm3(dec_lookup)
        dec_lookup = dec_lookup + self.dropout3(self.feed_forward(declookup_norm))
        if use_cache:
            return dec_lookup, (present_self, present_cross)
        return dec_lookup


class Encoder(nn.Module):
    def __init__(
        self,
        enc_voc_size,
        max_src_len,
        model_dim,
        hidden_dim,
        n_head,
        n_layer,
        drop_prob,
        device,
        use_bias=True,
    ):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(model_dim=model_dim,
                                              max_len=max_src_len,
                                              vocab_size=enc_voc_size,
                                              drop_prob=drop_prob,
                                              device=device
                                              )
        self.layers = nn.ModuleList([EncoderLayer(model_dim,
                                                  hidden_dim,
                                                  n_head,
                                                  drop_prob,
                                                  device=device,
                                                  use_bias=use_bias) for _ in range(n_layer)])
        # add norm layer at the end
        self.norm = LayerNorm(model_dim, device=device, use_bias=use_bias)
    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size,
        max_trg_len,
        model_dim,
        hidden_dim,
        n_head,
        n_layer,
        drop_prob,
        device,
        use_bias=True,
    ):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(model_dim=model_dim,
                                              max_len=max_trg_len,
                                              vocab_size=dec_voc_size,
                                              drop_prob=drop_prob,
                                              device=device)
        self.layers = nn.ModuleList([DecoderLayer(model_dim,
                                                  hidden_dim,
                                                  n_head,
                                                  drop_prob,
                                                  device=device,
                                                  use_bias=use_bias) for _ in range(n_layer)])
        self.norm = LayerNorm(model_dim, device=device, use_bias=use_bias)

    def forward(
        self,
        trg,
        enc_src,
        src_mask,
        trg_mask,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        if past_key_values is None:
            past_key_values = (None,) * len(self.layers)
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must match decoder layers")

        trg = self.embedding(trg, position_ids=position_ids)
        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            layer_output = layer(
                trg,
                enc_src,
                src_mask,
                trg_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                trg, present_key_value = layer_output
                present_key_values.append(present_key_value)
            else:
                trg = layer_output
        trg = self.norm(trg)  # add another norm layer before linear classification layer
        if use_cache:
            return trg, tuple(present_key_values)
        return trg


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab, use_bias=True):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab, bias=use_bias)

    def forward(self, x):
        return self.proj(x)
        # return log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, generator, model_dim, n_head,
                 max_src_len, max_trg_len, hidden_dim, n_layer, drop_prob, src_pad_idx, trg_pad_idx,
                 src_bos_idx, trg_bos_idx, src_eos_idx, trg_eos_idx, device,
                 use_bias=True, use_weight_tying=True):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_bos_idx = src_bos_idx
        self.trg_bos_idx = trg_bos_idx
        self.src_eos_idx = src_eos_idx
        self.trg_eos_idx = trg_eos_idx
        self.enc_voc_size = enc_voc_size
        self.dec_voc_size = dec_voc_size
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.encoder = Encoder(model_dim=model_dim,
                               n_head=n_head,
                               max_src_len=max_src_len,
                               hidden_dim=hidden_dim,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layer=n_layer,
                               device=device,
                               use_bias=use_bias)
        self.decoder = Decoder(model_dim=model_dim,
                               n_head=n_head,
                               max_trg_len=max_trg_len,
                               hidden_dim=hidden_dim,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layer=n_layer,
                               device=device,
                               use_bias=use_bias)
        self.generator = generator
        self.device = device
        self.use_weight_tying = use_weight_tying
        if self.use_weight_tying:
            self.generator.proj.weight = self.decoder.embedding.tok_embedding.weight

    @staticmethod
    def make_pad_mask(tensor, pad_idx, device=None):
        pad_mask = (tensor != pad_idx).unsqueeze(-2)
        if device:
            pad_mask = pad_mask.to(device)
        return pad_mask

    @staticmethod
    def make_no_peak_mask(tensor, device=None):
        # tensor.size(1)
        mask = torch.triu(torch.ones(1, tensor.size(1), tensor.size(1)), diagonal=1).type(torch.uint8)
        if device:
            mask = mask.to(device)
        return mask == 0

    def forward(self, src, trg):
        # there are many coding style to generate mask
        src_mask = self.make_pad_mask(src, self.src_pad_idx, self.device)
        trg_mask = self.make_pad_mask(trg, self.trg_pad_idx,
                                      self.device) & self.make_no_peak_mask(trg, self.device)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)
        output = self.generator(output)
        return output


# ========== GPT with RoPE ==========

class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) - supports position_ids for left padding"""
    def __init__(self, head_dim, max_seq_len=2048, base=10000, device=None):
        super(RoPE, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device

        # Precompute frequency matrix: [head_dim // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        if device:
            inv_freq = inv_freq.to(device)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, position_ids=None):
        """
        Apply RoPE to input tensor x
        x: [batch_size, n_head, seq_len, head_dim]
        position_ids: [batch_size, seq_len] - optional, for left padding support
        """
        batch_size, n_head, seq_len, head_dim = x.size()

        if position_ids is None:
            # Default: sequential positions [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            # Compute frequencies: [seq_len, head_dim // 2]
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            # Expand to match x shape: [1, 1, seq_len, head_dim // 2]
            cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
            sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        else:
            # Use provided position_ids for left padding support
            # position_ids: [batch_size, seq_len]
            position_ids = position_ids.float()
            # Compute frequencies for each position: [batch_size, seq_len, head_dim // 2]
            freqs = position_ids.unsqueeze(-1) * self.inv_freq.to(x.device).unsqueeze(0).unsqueeze(0)
            # [batch_size, seq_len, head_dim // 2]
            cos = torch.cos(freqs).unsqueeze(1)  # [batch_size, 1, seq_len, head_dim // 2]
            sin = torch.sin(freqs).unsqueeze(1)  # [batch_size, 1, seq_len, head_dim // 2]

        # Split x into two halves along the last dimension
        x1, x2 = x.chunk(2, dim=-1)  # Each: [batch_size, n_head, seq_len, head_dim // 2]

        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated


class MultiHeadAttentionWithRoPE(nn.Module):
    """RoPE attention with optional grouped-query attention and learned sinks."""
    def __init__(self, embed_dim, n_head, drop_prob, device, max_seq_len=2048,
                 rope_base=10000, n_kv_head=None, attention_sink_size=0,
                 use_rope=True, use_bias=True):
        super(MultiHeadAttentionWithRoPE, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_kv_head = n_head if n_kv_head is None else n_kv_head
        self.attention_sink_size = attention_sink_size
        self.use_rope = use_rope
        self.head_dim = self.embed_dim // self.n_head
        assert self.embed_dim % self.n_head == 0, "embed_dim must be divisible by n_head"
        assert self.n_kv_head > 0, "n_kv_head must be positive"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert self.attention_sink_size >= 0, "attention_sink_size cannot be negative"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.num_key_value_groups = self.n_head // self.n_kv_head

        self.linear_q = nn.Linear(
            self.embed_dim, self.head_dim * self.n_head, bias=use_bias, device=device
        )
        self.linear_k = nn.Linear(
            self.embed_dim, self.head_dim * self.n_kv_head, bias=use_bias, device=device
        )
        self.linear_v = nn.Linear(
            self.embed_dim, self.head_dim * self.n_kv_head, bias=use_bias, device=device
        )
        self.scaled_dot_product_attention = ScaleDotProductAttention()
        self.linear_attention = nn.Linear(
            self.head_dim * self.n_head, self.embed_dim, bias=use_bias, device=device
        )
        self.dropout = nn.Dropout(drop_prob)
        self.rope = (
            RoPE(self.head_dim, max_seq_len=max_seq_len, base=rope_base, device=device)
            if self.use_rope
            else None
        )

        if self.attention_sink_size:
            sink_shape = (1, self.n_kv_head, self.attention_sink_size, self.head_dim)
            self.sink_key = nn.Parameter(torch.empty(sink_shape, device=device))
            self.sink_value = nn.Parameter(torch.empty(sink_shape, device=device))
            nn.init.normal_(self.sink_key, mean=0.0, std=self.head_dim ** -0.5)
            nn.init.normal_(self.sink_value, mean=0.0, std=self.head_dim ** -0.5)
        else:
            self.register_parameter("sink_key", None)
            self.register_parameter("sink_value", None)

    def _repeat_key_value(self, hidden_states):
        if self.num_key_value_groups == 1:
            return hidden_states
        return hidden_states.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        """
        Args:
            q, k, v: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len, seq_len] or None
            position_ids: [batch_size, seq_len] or None - for left padding support
            past_key_value: optional tuple of cached K/V before head repetition
            use_cache: return the updated K/V cache when True
        """
        batch_size = q.size(0)
        query_len = q.size(1)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        Q = q.view(
            batch_size, query_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        K = k.view(
            batch_size, query_len, self.n_kv_head, self.head_dim
        ).transpose(1, 2)
        V = v.view(
            batch_size, query_len, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        if self.rope is not None:
            Q = self.rope(Q, position_ids)
            K = self.rope(K, position_ids)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            K = torch.cat((past_key, K), dim=2)
            V = torch.cat((past_value, V), dim=2)

        present_key_value = (K, V) if use_cache else None

        if self.attention_sink_size:
            sink_key = self.sink_key.expand(batch_size, -1, -1, -1)
            sink_value = self.sink_value.expand(batch_size, -1, -1, -1)
            K = torch.cat((sink_key, K), dim=2)
            V = torch.cat((sink_value, V), dim=2)
            if mask is not None:
                sink_mask = torch.ones(
                    batch_size,
                    mask.size(-2),
                    self.attention_sink_size,
                    dtype=torch.bool,
                    device=mask.device,
                )
                mask = torch.cat((sink_mask, mask.bool()), dim=-1)

        K = self._repeat_key_value(K)
        V = self._repeat_key_value(V)

        if mask is not None:
            mask = mask.unsqueeze(1)

        context, _ = self.scaled_dot_product_attention(Q, K, V, self.dropout, mask)

        output = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.head_dim * self.n_head
        )
        output = self.linear_attention(output)
        if use_cache:
            return output, present_key_value
        return output


class GPTLayer(nn.Module):
    """Configurable decoder-only Transformer layer."""
    def __init__(self, model_dim, hidden_dim, n_head, drop_prob, device, max_seq_len=2048,
                 rope_base=10000, n_kv_head=None, attention_sink_size=0,
                 use_rope=True, use_swiglu=True, use_rms_norm=True,
                 use_pre_norm=True, use_bias=True):
        super(GPTLayer, self).__init__()
        self.use_pre_norm = use_pre_norm
        self.self_attention = MultiHeadAttentionWithRoPE(
            model_dim,
            n_head,
            drop_prob,
            device,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            n_kv_head=n_kv_head,
            attention_sink_size=attention_sink_size,
            use_rope=use_rope,
            use_bias=use_bias,
        )
        if use_rms_norm:
            self.layer_norm1 = RMSNorm(model_dim, device=device)
        else:
            self.layer_norm1 = LayerNorm(model_dim, device=device, use_bias=use_bias)
        self.dropout1 = nn.Dropout(drop_prob)
        if use_swiglu:
            self.feed_forward = SwiGLU(
                model_dim, hidden_dim, drop_prob, device=device
            )
        else:
            self.feed_forward = PositionWiseFeedForward(
                model_dim, hidden_dim, drop_prob, device=device, use_bias=use_bias
            )
        if use_rms_norm:
            self.layer_norm2 = RMSNorm(model_dim, device=device)
        else:
            self.layer_norm2 = LayerNorm(model_dim, device=device, use_bias=use_bias)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(
        self,
        x,
        mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        """
        Args:
            x: [batch_size, seq_len, model_dim]
            mask: [batch_size, seq_len, seq_len] or None
            position_ids: [batch_size, seq_len] or None - for left padding support
        """
        attention_input = self.layer_norm1(x) if self.use_pre_norm else x
        attention_output = self.self_attention(
            attention_input,
            attention_input,
            attention_input,
            mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            attention_output, present_key_value = attention_output

        if self.use_pre_norm:
            x = x + self.dropout1(attention_output)
            xnorm = self.layer_norm2(x)
            x = x + self.dropout2(self.feed_forward(xnorm))
        else:
            x = self.layer_norm1(x + self.dropout1(attention_output))
            x = self.layer_norm2(x + self.dropout2(self.feed_forward(x)))
        if use_cache:
            return x, present_key_value
        return x


class GPTEmbedding(nn.Module):
    """Token embedding with optional sinusoidal positions when RoPE is disabled."""
    def __init__(self, vocab_size, model_dim, max_context_len, drop_prob, device, use_rope=True):
        super(GPTEmbedding, self).__init__()
        self.model_dim = model_dim
        self.use_rope = use_rope
        self.tok_embedding = nn.Embedding(vocab_size, model_dim, device=device)
        self.pos_embedding = None if use_rope else PositionLayer(max_context_len, model_dim, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, position_ids=None):
        tok_embedding = self.tok_embedding(x) * math.sqrt(self.model_dim)
        if self.pos_embedding is not None:
            if position_ids is None:
                pos_embedding = self.pos_embedding(x)
            else:
                pos_embedding = self.pos_embedding.pe[0, position_ids]
            tok_embedding = tok_embedding + pos_embedding
        return self.dropout(tok_embedding)


class GPT(nn.Module):
    """GPT Model (decoder-only architecture with RoPE) - supports attention_mask and position_ids"""
    def __init__(self, vocab_size, model_dim, n_head, max_context_len, hidden_dim, n_layer,
                 drop_prob, pad_idx, bos_idx, eos_idx, device, rope_base=10000,
                 n_kv_head=None, attention_sink_size=0, use_rope=True,
                 use_swiglu=True, use_rms_norm=True, use_pre_norm=True,
                 use_bias=True, use_weight_tying=True):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.n_head = n_head
        self.n_kv_head = n_head if n_kv_head is None else n_kv_head
        self.attention_sink_size = attention_sink_size
        self.use_rope = use_rope
        self.max_context_len = max_context_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device
        self.use_weight_tying = use_weight_tying

        self.embedding = GPTEmbedding(
            vocab_size,
            model_dim,
            max_context_len,
            drop_prob,
            device,
            use_rope=use_rope,
        )
        self.layers = nn.ModuleList([
            GPTLayer(
                model_dim,
                hidden_dim,
                n_head,
                drop_prob,
                device,
                max_seq_len=max_context_len,
                rope_base=rope_base,
                n_kv_head=self.n_kv_head,
                attention_sink_size=self.attention_sink_size,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
                use_rms_norm=use_rms_norm,
                use_pre_norm=use_pre_norm,
                use_bias=use_bias,
            )
            for _ in range(n_layer)
        ])
        if use_pre_norm:
            self.norm = (
                RMSNorm(model_dim, device=device)
                if use_rms_norm
                else LayerNorm(model_dim, device=device, use_bias=use_bias)
            )
        else:
            self.norm = nn.Identity()
        self.generator = Generator(model_dim, vocab_size, use_bias=use_bias)
        if self.use_weight_tying:
            self.generator.proj.weight = self.embedding.tok_embedding.weight

    @staticmethod
    def make_pad_mask(tensor, pad_idx, device=None):
        """Create padding mask from input_ids: [batch_size, seq_len, seq_len]"""
        batch_size, seq_len = tensor.size()
        pad_mask = (tensor != pad_idx)  # [batch_size, seq_len]
        pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]
        if device:
            pad_mask = pad_mask.to(device)
        return pad_mask

    @staticmethod
    def make_pad_mask_from_attention_mask(attention_mask, device=None):
        """Create padding mask from attention_mask: [batch_size, seq_len, seq_len]
        attention_mask: [batch_size, seq_len] with 1 for real tokens, 0 for padding
        """
        batch_size, seq_len = attention_mask.size()
        # Expand to [batch_size, seq_len, seq_len]
        pad_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).bool()
        if device:
            pad_mask = pad_mask.to(device)
        return pad_mask

    @staticmethod
    def make_causal_mask(seq_len, device=None):
        """Create causal mask (no-peak mask) for decoder-only model: [seq_len, seq_len]"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return ~mask  # [seq_len, seq_len]

    @staticmethod
    def make_position_ids_from_attention_mask(attention_mask):
        """
        Create position_ids from attention_mask for left padding.
        For left padding, position should start from 0 for the first real token.

        Example:
            attention_mask: [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
            position_ids:   [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]

        Args:
            attention_mask: [batch_size, seq_len] with 1 for real tokens, 0 for padding
        Returns:
            position_ids: [batch_size, seq_len]
        """
        # cumsum along sequence dimension gives us incrementing positions
        # subtract 1 because cumsum starts from 1, we want it to start from 0
        position_ids = attention_mask.long().cumsum(-1) - 1
        # Clamp to 0 for padding positions (they had cumsum 0, now -1)
        position_ids = position_ids.clamp(min=0)
        return position_ids

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Forward pass for GPT with support for attention_mask and position_ids

        Args:
            input_ids: [batch_size, seq_len] - input token indices
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding (optional)
            position_ids: positions for the current input tokens
            past_key_values: cached K/V tuple for each Transformer layer
            use_cache: return updated per-layer K/V cache when True

        Returns:
            logits, or (logits, present_key_values) when use_cache=True
        """
        batch_size, query_len = input_ids.size()
        if past_key_values is None:
            past_key_values = (None,) * len(self.layers)
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must match the number of layers")

        first_past = past_key_values[0]
        past_len = 0 if first_past is None else first_past[0].size(2)
        key_len = past_len + query_len

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                key_len,
                dtype=torch.long,
                device=input_ids.device,
            )
        elif attention_mask.size(1) != key_len:
            raise ValueError(
                "attention_mask length must equal cached tokens plus input tokens"
            )
        else:
            attention_mask = attention_mask.to(input_ids.device)

        if position_ids is None:
            all_position_ids = self.make_position_ids_from_attention_mask(
                attention_mask
            )
            position_ids = all_position_ids[:, -query_len:]
        else:
            position_ids = position_ids.to(input_ids.device)

        key_positions = torch.arange(key_len, device=input_ids.device)
        query_positions = torch.arange(
            past_len, key_len, device=input_ids.device
        ).unsqueeze(-1)
        causal_mask = key_positions.unsqueeze(0) <= query_positions
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        pad_mask = attention_mask.bool().unsqueeze(1).expand(
            -1, query_len, -1
        )
        mask = pad_mask & causal_mask

        # Embedding
        x = self.embedding(input_ids, position_ids=position_ids)

        # Transformer layers
        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            layer_output = layer(
                x,
                mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                x, present_key_value = layer_output
                present_key_values.append(present_key_value)
            else:
                x = layer_output

        # Final norm
        x = self.norm(x)

        # Generate logits
        output = self.generator(x)
        if use_cache:
            return output, tuple(present_key_values)
        return output


def _resolve_gpt_kv_heads(configs):
    n_head = int(configs["n_head"])
    if not configs.get("use_gqa", False):
        return n_head

    if "n_kv_head" not in configs:
        raise ValueError(
            "use_gqa=True requires n_kv_head to be set below n_head"
        )

    n_kv_head = int(configs["n_kv_head"])
    if not 0 < n_kv_head < n_head:
        raise ValueError(
            "use_gqa=True requires 0 < n_kv_head < n_head; "
            f"got n_kv_head={n_kv_head}, n_head={n_head}"
        )
    if n_head % n_kv_head != 0:
        raise ValueError(
            "use_gqa=True requires n_head to be divisible by n_kv_head; "
            f"got n_head={n_head}, n_kv_head={n_kv_head}"
        )
    return n_kv_head


def build_GPT(unified_tokenizer, max_context_len, configs: dict):
    """
    Build GPT model with RoPE
    unified_tokenizer: shared source/target tokenizer or vocabulary size
    max_context_len: maximum context length
    configs: configuration dictionary
    """
    use_attention_sink = configs.get('use_attention_sink', True)
    n_kv_head = _resolve_gpt_kv_heads(configs)
    attention_sink_size = configs.get('attention_sink_size', 0) if use_attention_sink else 0
    use_bias = configs.get('use_bias', True)
    use_weight_tying = configs.get('use_weight_tying', True)

    model = GPT(
        vocab_size=_vocab_size(unified_tokenizer),
        model_dim=configs['model_dim'],
        n_head=configs['n_head'],
        max_context_len=max_context_len,
        hidden_dim=configs['hidden_dim'],
        n_layer=configs['n_layer'],
        drop_prob=configs['drop_prob'],
        pad_idx=configs['pad_token'],
        bos_idx=configs['bos_token'],
        eos_idx=configs['eos_token'],
        device=_resolve_device(configs),
        rope_base=configs.get('rope_base', 10000),
        n_kv_head=n_kv_head,
        attention_sink_size=attention_sink_size,
        use_rope=configs.get('use_rope', True),
        use_swiglu=configs.get('use_swiglu', True),
        use_rms_norm=configs.get('use_rms_norm', True),
        use_pre_norm=configs.get('use_pre_norm', True),
        use_bias=use_bias,
        use_weight_tying=use_weight_tying,
    )
    return model


# class VisionPatchEmbedding(nn.Module):
#     """Convert an image into patch tokens with learned 2D grid positions."""

#     def __init__(self, image_size, patch_size, in_channels, model_dim, device):
#         super().__init__()
#         if image_size % patch_size != 0:
#             raise ValueError("image_size must be divisible by patch_size")
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.grid_size = image_size // patch_size
#         self.num_patches = self.grid_size ** 2
#         self.projection = nn.Conv2d(
#             in_channels,
#             model_dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#             device=device,
#         )
#         self.position_embedding = nn.Parameter(
#             torch.zeros(1, self.num_patches, model_dim, device=device)
#         )
#         nn.init.normal_(self.position_embedding, mean=0.0, std=model_dim ** -0.5)

#     def forward(self, images):
#         if images.shape[-2:] != (self.image_size, self.image_size):
#             raise ValueError(
#                 f"expected images with size {self.image_size}x{self.image_size}, "
#                 f"got {tuple(images.shape[-2:])}"
#             )
#         patches = self.projection(images).flatten(2).transpose(1, 2)
#         return patches + self.position_embedding


# class VisionLanguageModel(nn.Module):
#     """Single-stream VLM using [text prefix | image patches | text suffix]."""

#     def __init__(self, vocab_size, model_dim, n_head, max_len, hidden_dim, n_layer,
#                  drop_prob, pad_idx, bos_idx, eos_idx, device, image_size=224,
#                  patch_size=16, image_channels=3, rope_base=10000,
#                  n_kv_head=None, attention_sink_size=0, use_rope=True,
#                  use_swiglu=True, use_rms_norm=True, use_pre_norm=True):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.model_dim = model_dim
#         self.max_len = max_len
#         self.pad_idx = pad_idx
#         self.bos_idx = bos_idx
#         self.eos_idx = eos_idx
#         self.device = device
#         self.text_embedding = nn.Embedding(vocab_size, model_dim, device=device)
#         self.vision_embedding = VisionPatchEmbedding(
#             image_size, patch_size, image_channels, model_dim, device
#         )
#         self.position_embedding = None if use_rope else PositionLayer(max_len, model_dim, device)
#         self.embedding_dropout = nn.Dropout(drop_prob)
#         kv_heads = n_head if n_kv_head is None else n_kv_head
#         self.layers = nn.ModuleList([
#             GPTLayer(
#                 model_dim,
#                 hidden_dim,
#                 n_head,
#                 drop_prob,
#                 device,
#                 max_seq_len=max_len,
#                 rope_base=rope_base,
#                 n_kv_head=kv_heads,
#                 attention_sink_size=attention_sink_size,
#                 use_rope=use_rope,
#                 use_swiglu=use_swiglu,
#                 use_rms_norm=use_rms_norm,
#                 use_pre_norm=use_pre_norm,
#             )
#             for _ in range(n_layer)
#         ])
#         if use_pre_norm:
#             self.norm = (
#                 RMSNorm(model_dim, device=device)
#                 if use_rms_norm
#                 else LayerNorm(model_dim, device=device)
#             )
#         else:
#             self.norm = nn.Identity()
#         self.generator = Generator(model_dim, vocab_size)

#     @staticmethod
#     def make_block_causal_mask(prefix_length, image_length, suffix_length,
#                                batch_size, device, key_attention_mask=None):
#         seq_len = prefix_length + image_length + suffix_length
#         mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
#         image_start = prefix_length
#         image_end = image_start + image_length
#         mask[image_start:image_end, image_start:image_end] = True
#         mask = mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
#         if key_attention_mask is not None:
#             expected_shape = (batch_size, seq_len)
#             if key_attention_mask.shape != expected_shape:
#                 raise ValueError(
#                     f"key_attention_mask must have shape {expected_shape}, "
#                     f"got {tuple(key_attention_mask.shape)}"
#                 )
#             mask = mask & key_attention_mask.bool().unsqueeze(1)
#         return mask

#     @staticmethod
#     def make_position_ids(attention_mask):
#         return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

#     @staticmethod
#     def make_labels(prefix_length, image_length, suffix_input_ids,
#                     ignore_index=0, pad_idx=0):
#         batch_size, suffix_length = suffix_input_ids.shape
#         seq_len = prefix_length + image_length + suffix_length
#         labels = torch.full(
#             (batch_size, seq_len),
#             ignore_index,
#             dtype=suffix_input_ids.dtype,
#             device=suffix_input_ids.device,
#         )
#         answer_start = prefix_length + image_length - 1
#         answer_targets = suffix_input_ids.masked_fill(
#             suffix_input_ids == pad_idx, ignore_index
#         )
#         labels[:, answer_start:answer_start + suffix_length] = answer_targets
#         return labels

#     def forward(self, prefix_input_ids, images, suffix_input_ids=None,
#                 prefix_attention_mask=None, suffix_attention_mask=None,
#                 position_ids=None):
#         batch_size, prefix_length = prefix_input_ids.shape
#         if images.shape[0] != batch_size:
#             raise ValueError("text and image batch sizes must match")

#         prefix_embeddings = self.text_embedding(prefix_input_ids) * math.sqrt(self.model_dim)
#         image_embeddings = self.vision_embedding(images)
#         image_length = image_embeddings.shape[1]
#         embedding_parts = [prefix_embeddings, image_embeddings]

#         suffix_length = 0
#         if suffix_input_ids is not None:
#             if suffix_input_ids.shape[0] != batch_size:
#                 raise ValueError("prefix and suffix batch sizes must match")
#             suffix_length = suffix_input_ids.shape[1]
#             embedding_parts.append(
#                 self.text_embedding(suffix_input_ids) * math.sqrt(self.model_dim)
#             )

#         hidden_states = torch.cat(embedding_parts, dim=1)
#         if hidden_states.shape[1] > self.max_len:
#             raise ValueError(
#                 f"multimodal sequence length {hidden_states.shape[1]} "
#                 f"exceeds max_len {self.max_len}"
#             )

#         if prefix_attention_mask is None:
#             prefix_attention_mask = (prefix_input_ids != self.pad_idx).long()
#         image_attention_mask = torch.ones(
#             batch_size, image_length, dtype=torch.long, device=images.device
#         )
#         attention_parts = [prefix_attention_mask.to(images.device), image_attention_mask]
#         if suffix_length:
#             if suffix_attention_mask is None:
#                 suffix_attention_mask = (suffix_input_ids != self.pad_idx).long()
#             attention_parts.append(suffix_attention_mask.to(images.device))
#         attention_mask = torch.cat(attention_parts, dim=1)

#         block_causal_mask = self.make_block_causal_mask(
#             prefix_length,
#             image_length,
#             suffix_length,
#             batch_size,
#             images.device,
#             key_attention_mask=attention_mask,
#         )
#         if position_ids is None:
#             position_ids = self.make_position_ids(attention_mask)
#         if self.position_embedding is not None:
#             hidden_states = hidden_states + self.position_embedding(attention_mask)
#         hidden_states = self.embedding_dropout(hidden_states)

#         for layer in self.layers:
#             hidden_states = layer(
#                 hidden_states,
#                 mask=block_causal_mask,
#                 position_ids=position_ids,
#             )
#         return self.generator(self.norm(hidden_states))
