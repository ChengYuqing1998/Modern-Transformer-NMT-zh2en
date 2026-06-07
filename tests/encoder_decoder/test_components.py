import unittest

import torch

from models.transformer import Transformer, build_model
from tokenizer.tokenizer import Vocabulary


class EncoderDecoderComponentTest(unittest.TestCase):
    def test_forward_and_backward(self):
        source = Vocabulary("zh")
        target = Vocabulary("en")
        source.add_text("你好", character_level=True)
        target.add_text("hello world")
        config = {
            "model_dim": 16,
            "hidden_dim": 32,
            "n_layer": 2,
            "n_head": 4,
            "drop_prob": 0.0,
            "pad_token": 0,
            "bos_token": 1,
            "eos_token": 2,
            "_device": "cpu",
        }
        model = build_model(source, target, 16, 16, config)
        source_ids = torch.tensor([[1, 4, 5, 2, 0]])
        target_ids = torch.tensor([[1, 4, 5, 2]])

        logits = model(source_ids, target_ids)
        logits.sum().backward()

        self.assertEqual(logits.shape, (1, 4, len(target)))
        self.assertIsNotNone(model.encoder.embedding.tok_embedding.weight.grad)
        self.assertIs(
            model.decoder.embedding.tok_embedding.weight,
            model.generator.proj.weight,
        )
        self.assertEqual(model.max_src_len, 16)
        self.assertEqual(model.max_trg_len, 16)

    def test_masks(self):
        inputs = torch.tensor([[1, 4, 0]])
        pad_mask = Transformer.make_pad_mask(inputs, 0)
        causal_mask = Transformer.make_no_peak_mask(inputs)

        self.assertEqual(pad_mask.tolist(), [[[True, True, False]]])
        self.assertFalse(causal_mask[0, 0, 1])
        self.assertTrue(causal_mask[0, 2, 1])


if __name__ == "__main__":
    unittest.main()
