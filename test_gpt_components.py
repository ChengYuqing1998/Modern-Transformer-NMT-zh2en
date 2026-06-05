import unittest

import torch

from transformer import (
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
            max_len=16,
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
            max_len=16,
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


if __name__ == "__main__":
    unittest.main()
