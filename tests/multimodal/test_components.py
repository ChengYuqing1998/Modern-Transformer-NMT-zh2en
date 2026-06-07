import unittest

import torch

from models.transformer import VisionLanguageModel, VisionPatchEmbedding


class VLMComponentTest(unittest.TestCase):
    def test_patch_embedding(self):
        embedding = VisionPatchEmbedding(
            image_size=8,
            patch_size=4,
            in_channels=3,
            model_dim=16,
            device=torch.device("cpu"),
        )
        images = torch.randn(2, 3, 8, 8)

        output = embedding(images)

        self.assertEqual(output.shape, (2, 4, 16))

    def test_block_causal_mask(self):
        mask = VisionLanguageModel.make_block_causal_mask(
            prefix_length=2,
            image_length=3,
            suffix_length=2,
            batch_size=1,
            device=torch.device("cpu"),
        )[0]

        # Prefix text cannot see future text or image tokens.
        self.assertFalse(mask[0, 1])
        self.assertFalse(mask[1, 2])

        # Every image patch sees past prefix text and the full image block.
        self.assertTrue(mask[2, 0])
        self.assertTrue(mask[2, 4])
        self.assertTrue(mask[4, 2])

        # Image tokens cannot see future answer text.
        self.assertFalse(mask[4, 5])

        # Answer text sees all previous text and image tokens, but not future text.
        self.assertTrue(mask[5, 4])
        self.assertFalse(mask[5, 6])
        self.assertTrue(mask[6, 5])

    def test_vlm_forward_and_backward(self):
        model = VisionLanguageModel(
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
            image_size=8,
            patch_size=4,
            image_channels=3,
        )
        prefix = torch.tensor([[1, 4], [1, 5]])
        suffix = torch.tensor([[6, 7], [8, 2]])
        images = torch.randn(2, 3, 8, 8)

        logits = model(prefix, images, suffix)
        logits.sum().backward()

        self.assertEqual(logits.shape, (2, 8, 32))
        self.assertIsNotNone(model.vision_embedding.projection.weight.grad)

    def test_vlm_labels_ignore_prefix_and_image(self):
        suffix = torch.tensor([[7, 8, 2], [9, 2, 0]])

        labels = VisionLanguageModel.make_labels(
            prefix_length=2,
            image_length=4,
            suffix_input_ids=suffix,
            ignore_index=-100,
            pad_idx=0,
        )

        self.assertEqual(labels.shape, (2, 9))
        self.assertTrue(torch.all(labels[:, :5] == -100))
        self.assertEqual(labels[0, 5:].tolist(), [7, 8, 2, -100])
        self.assertEqual(labels[1, 5:].tolist(), [9, 2, -100, -100])


if __name__ == "__main__":
    unittest.main()
