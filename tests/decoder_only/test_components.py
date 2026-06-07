import unittest

import torch

from models.transformer import (
    GPT,
    LayerNorm,
    MultiHeadAttentionWithRoPE,
    PositionWiseFeedForward,
    SwiGLU,
)


class GPTComponentTest(unittest.TestCase):
    def test_swiglu_shape_and_backward(self):
        layer = SwiGLU(16, 32, 0.0, torch.device("cpu"))
        inputs = torch.randn(2, 5, 16, requires_grad=True)

        output = layer(inputs)
        output.sum().backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNotNone(inputs.grad)

    def test_gqa_with_attention_sinks(self):
        attention = MultiHeadAttentionWithRoPE(
            embed_dim=16,
            n_head=4,
            n_kv_head=2,
            attention_sink_size=2,
            drop_prob=0.0,
            device=torch.device("cpu"),
        )
        inputs = torch.randn(2, 5, 16, requires_grad=True)
        mask = torch.tril(torch.ones(2, 5, 5, dtype=torch.bool))

        output = attention(inputs, inputs, inputs, mask=mask)
        output.sum().backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(attention.linear_k.out_features, 8)
        self.assertEqual(attention.sink_key.shape, (1, 2, 2, 4))
        self.assertIsNotNone(attention.sink_key.grad)

    def test_gpt_supports_left_padding_with_gqa_and_sinks(self):
        model = GPT(
            vocab_size=32,
            model_dim=16,
            n_head=4,
            n_kv_head=2,
            attention_sink_size=2,
            max_context_len=16,
            hidden_dim=32,
            n_layer=2,
            drop_prob=0.0,
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            device=torch.device("cpu"),
        )
        input_ids = torch.tensor([[0, 0, 1, 4, 5], [0, 1, 6, 7, 8]])
        attention_mask = (input_ids != 0).long()

        output = model(input_ids, attention_mask=attention_mask)
        output.sum().backward()

        self.assertEqual(output.shape, (2, 5, 32))

    def test_classic_decoder_configuration(self):
        model = GPT(
            vocab_size=32,
            model_dim=16,
            n_head=4,
            n_kv_head=4,
            attention_sink_size=0,
            use_rope=False,
            use_swiglu=False,
            use_rms_norm=False,
            use_pre_norm=False,
            max_context_len=16,
            hidden_dim=32,
            n_layer=2,
            drop_prob=0.0,
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            device=torch.device("cpu"),
        )
        input_ids = torch.tensor([[1, 4, 5, 0], [1, 6, 7, 8]])

        output = model(input_ids)
        output.sum().backward()

        self.assertEqual(output.shape, (2, 4, 32))
        self.assertIsNone(model.layers[0].self_attention.rope)
        self.assertIsInstance(model.layers[0].feed_forward, PositionWiseFeedForward)
        self.assertIsInstance(model.layers[0].layer_norm1, LayerNorm)
        self.assertIsInstance(model.norm, torch.nn.Identity)
        self.assertIs(model.embedding.tok_embedding.weight, model.generator.proj.weight)

    def test_kv_cache_matches_full_prefix_logits(self):
        torch.manual_seed(0)
        model = GPT(
            vocab_size=32,
            model_dim=16,
            n_head=4,
            n_kv_head=2,
            attention_sink_size=2,
            max_context_len=16,
            hidden_dim=32,
            n_layer=2,
            drop_prob=0.0,
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            device=torch.device("cpu"),
        ).eval()
        prompt = torch.tensor([[1, 4, 5]])
        attention_mask = torch.ones_like(prompt)

        with torch.no_grad():
            prefill_logits, cache = model(
                prompt,
                attention_mask=attention_mask,
                use_cache=True,
            )
            next_token = prefill_logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            full_sequence = torch.cat((prompt, next_token), dim=1)
            full_mask = torch.ones_like(full_sequence)

            full_logits = model(
                full_sequence,
                attention_mask=full_mask,
            )
            cached_logits, updated_cache = model(
                next_token,
                attention_mask=full_mask,
                past_key_values=cache,
                use_cache=True,
            )

        torch.testing.assert_close(
            cached_logits[:, -1, :],
            full_logits[:, -1, :],
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[0][0].shape, (1, 2, 3, 4))
        self.assertEqual(updated_cache[0][0].shape, (1, 2, 4, 4))

    def test_kv_cache_supports_sinusoidal_positions(self):
        torch.manual_seed(0)
        model = GPT(
            vocab_size=32,
            model_dim=16,
            n_head=4,
            n_kv_head=4,
            attention_sink_size=0,
            use_rope=False,
            use_swiglu=False,
            use_rms_norm=False,
            use_pre_norm=False,
            max_context_len=16,
            hidden_dim=32,
            n_layer=1,
            drop_prob=0.0,
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            device=torch.device("cpu"),
        ).eval()
        prompt = torch.tensor([[1, 4, 5]])
        prompt_mask = torch.ones_like(prompt)

        with torch.no_grad():
            prompt_logits, cache = model(
                prompt,
                attention_mask=prompt_mask,
                use_cache=True,
            )
            next_token = prompt_logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            full_sequence = torch.cat((prompt, next_token), dim=1)
            full_mask = torch.ones_like(full_sequence)
            full_logits = model(full_sequence, attention_mask=full_mask)
            cached_logits, _ = model(
                next_token,
                attention_mask=full_mask,
                past_key_values=cache,
                use_cache=True,
            )

        torch.testing.assert_close(
            cached_logits[:, -1, :],
            full_logits[:, -1, :],
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
