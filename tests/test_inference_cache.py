import unittest

import torch

from inference.translator import (
    DecoderOnlyTranslator,
    EncoderDecoderGenerationConfig,
    EncoderDecoderTranslator,
    GenerationConfig,
    GPTGenerator,
)
from models.transformer import GPT, build_model
from tokenizer.tokenizer import Vocabulary


class InferenceCacheTest(unittest.TestCase):
    def build_gpt(self):
        torch.manual_seed(0)
        return GPT(
            vocab_size=32,
            model_dim=16,
            n_head=4,
            n_kv_head=2,
            attention_sink_size=2,
            max_context_len=32,
            hidden_dim=32,
            n_layer=2,
            drop_prob=0.0,
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            device=torch.device("cpu"),
        ).eval()

    def build_decoder_only_translator(self):
        tokenizer = Vocabulary("mixed")
        tokenizer.add_text("你好世界", character_level=True)
        tokenizer.add_text("hello world good day")
        model = self.build_gpt()
        return DecoderOnlyTranslator(
            model,
            tokenizer,
            decoding_strategy="greedy",
            inference_max_new_tokens=5,
        )

    def build_encoder_decoder(self):
        torch.manual_seed(0)
        source = Vocabulary("zh")
        source.add_text("你好世界", character_level=True)
        target = Vocabulary("en")
        target.add_text("hello world good day")
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
        return build_model(source, target, 12, 12, config).eval()

    def test_decoder_only_greedy_cache_matches_full_prefix(self):
        generator = GPTGenerator(self.build_gpt())
        input_ids = torch.tensor([[0, 1, 4, 5], [1, 6, 7, 8]])
        attention_mask = (input_ids != 0).long()

        without_cache = generator.generate(
            input_ids,
            attention_mask,
            max_new_tokens=5,
            use_kv_cache=False,
        )
        with_cache = generator.generate(
            input_ids,
            attention_mask,
            max_new_tokens=5,
            use_kv_cache=True,
        )

        torch.testing.assert_close(with_cache, without_cache)

    def test_decoder_only_cache_only_forwards_latest_token_after_prefill(self):
        model = self.build_gpt()
        generator = GPTGenerator(model)
        input_ids = torch.tensor([[0, 1, 4, 5]])
        attention_mask = (input_ids != 0).long()
        forwarded_lengths = []
        original_forward = model.forward

        def tracked_forward(*args, **kwargs):
            current_input_ids = (
                kwargs["input_ids"] if "input_ids" in kwargs else args[0]
            )
            forwarded_lengths.append(current_input_ids.size(1))
            return original_forward(*args, **kwargs)

        model.forward = tracked_forward
        generator.generate(
            input_ids,
            attention_mask,
            max_new_tokens=4,
            use_kv_cache=True,
        )

        self.assertEqual(forwarded_lengths[0], input_ids.size(1))
        self.assertTrue(
            all(length == 1 for length in forwarded_lengths[1:])
        )

    def test_decoder_only_sampling_cache_matches_with_fixed_seed(self):
        generator = GPTGenerator(self.build_gpt())
        input_ids = torch.tensor([[0, 1, 4, 5], [1, 6, 7, 8]])
        attention_mask = (input_ids != 0).long()

        torch.manual_seed(123)
        without_cache = generator.generate(
            input_ids,
            attention_mask,
            max_new_tokens=5,
            do_sample=True,
            top_k=10,
            use_kv_cache=False,
        )
        torch.manual_seed(123)
        with_cache = generator.generate(
            input_ids,
            attention_mask,
            max_new_tokens=5,
            do_sample=True,
            top_k=10,
            use_kv_cache=True,
        )

        torch.testing.assert_close(with_cache, without_cache)

    def test_explicit_nucleus_sampling_strategy(self):
        generator = GPTGenerator(self.build_gpt())
        input_ids = torch.tensor([[1, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        torch.manual_seed(123)
        output = generator.generate(
            input_ids,
            attention_mask,
            decoding_strategy="nucleus_sampling",
            max_new_tokens=4,
            temperature=0.8,
            top_p=0.9,
            top_k=0,
            use_kv_cache=True,
        )

        self.assertEqual(output.shape, (1, 7))

    def test_decoding_strategy_validation(self):
        generator = GPTGenerator(self.build_gpt())
        input_ids = torch.tensor([[1, 4, 5]])

        with self.assertRaisesRegex(ValueError, "num_beams"):
            generator.generate(
                input_ids,
                decoding_strategy="beam_search",
                num_beams=1,
            )
        with self.assertRaisesRegex(ValueError, "top_p"):
            generator.generate(
                input_ids,
                decoding_strategy="nucleus_sampling",
                top_p=0.0,
            )

    def test_decoder_only_beam_cache_matches_full_prefix(self):
        generator = GPTGenerator(self.build_gpt())
        input_ids = torch.tensor([[0, 1, 4, 5], [1, 6, 7, 8]])
        attention_mask = (input_ids != 0).long()

        without_cache = generator.generate(
            input_ids,
            attention_mask,
            max_new_tokens=5,
            num_beams=3,
            use_kv_cache=False,
        )
        with_cache = generator.generate(
            input_ids,
            attention_mask,
            max_new_tokens=5,
            num_beams=3,
            use_kv_cache=True,
        )

        torch.testing.assert_close(with_cache, without_cache)

    def test_decoder_only_string_batch_preserves_batch_size(self):
        translator = self.build_decoder_only_translator()

        outputs = translator.translate(
            ["你", "你好世界"],
            max_new_tokens=3,
            decoding_strategy="greedy",
        )

        self.assertEqual(len(outputs), 2)

    def test_decoder_only_tensor_batch_preserves_batch_size(self):
        translator = self.build_decoder_only_translator()
        input_ids = torch.tensor([[0, 0, 1, 4], [1, 4, 5, 6]])
        attention_mask = (input_ids != 0).long()

        for strategy, num_beams in (
            ("greedy", 1),
            ("nucleus_sampling", 1),
            ("beam_search", 2),
        ):
            torch.manual_seed(123)
            outputs = translator.translate_batch(
                input_ids,
                attention_mask,
                max_new_tokens=3,
                decoding_strategy=strategy,
                num_beams=num_beams,
                top_p=0.9,
            )
            self.assertEqual(len(outputs), 2)

    def test_decoder_only_context_limit_uses_padded_batch_width(self):
        translator = self.build_decoder_only_translator()
        input_ids = torch.tensor([[0, 1, 4], [1, 4, 5]])
        attention_mask = (input_ids != 0).long()

        with self.assertRaisesRegex(ValueError, "max_context_len"):
            translator.translate_batch(
                input_ids,
                attention_mask,
                max_new_tokens=30,
                decoding_strategy="greedy",
            )

    def test_generation_config_is_not_mutated_by_generate(self):
        generator = GPTGenerator(self.build_gpt())
        config = GenerationConfig(
            max_new_tokens=2,
            decoding_strategy="greedy",
            num_beams=1,
        )

        generator.generate(
            torch.tensor([[1, 4, 5]]),
            generation_config=config,
            max_new_tokens=3,
        )

        self.assertEqual(config.max_new_tokens, 2)

    def test_encoder_decoder_beam_cache_matches_full_prefix(self):
        model = self.build_encoder_decoder()
        source_ids = torch.tensor(
            [[1, 4, 5, 2, 0], [1, 6, 7, 2, 0]]
        )

        without_cache = EncoderDecoderTranslator(
            model,
            beam_size=2,
            max_seq_len=8,
            use_kv_cache=False,
        ).translate(source_ids)
        with_cache = EncoderDecoderTranslator(
            model,
            beam_size=2,
            max_seq_len=8,
            use_kv_cache=True,
        ).translate(source_ids)

        self.assertEqual(with_cache, without_cache)

    def test_encoder_decoder_greedy_cache_matches_full_prefix(self):
        model = self.build_encoder_decoder()
        source_ids = torch.tensor(
            [[1, 4, 5, 2, 0], [1, 6, 7, 2, 0]]
        )

        without_cache = EncoderDecoderTranslator(
            model,
            max_seq_len=8,
            use_kv_cache=False,
        ).translate(
            source_ids,
            max_new_tokens=5,
            decoding_strategy="greedy",
        )
        with_cache = EncoderDecoderTranslator(
            model,
            max_seq_len=8,
            use_kv_cache=True,
        ).translate(
            source_ids,
            max_new_tokens=5,
            decoding_strategy="greedy",
        )

        self.assertEqual(with_cache, without_cache)

    def test_encoder_decoder_sampling_cache_matches_full_prefix(self):
        model = self.build_encoder_decoder()
        source_ids = torch.tensor(
            [[1, 4, 5, 2, 0], [1, 6, 7, 2, 0]]
        )

        torch.manual_seed(123)
        without_cache = EncoderDecoderTranslator(
            model,
            max_seq_len=8,
            use_kv_cache=False,
            decoding_strategy="nucleus_sampling",
        ).translate(
            source_ids,
            max_new_tokens=5,
            temperature=0.8,
            top_p=0.9,
            top_k=0,
        )
        torch.manual_seed(123)
        with_cache = EncoderDecoderTranslator(
            model,
            max_seq_len=8,
            use_kv_cache=True,
            decoding_strategy="nucleus_sampling",
        ).translate(
            source_ids,
            max_new_tokens=5,
            temperature=0.8,
            top_p=0.9,
            top_k=0,
        )

        self.assertEqual(with_cache, without_cache)

    def test_encoder_decoder_strategy_switch_does_not_mutate_defaults(self):
        model = self.build_encoder_decoder()
        source_ids = torch.tensor([[1, 4, 5, 2, 0]])
        translator = EncoderDecoderTranslator(
            model,
            beam_size=3,
            max_seq_len=8,
            use_kv_cache=True,
            decoding_strategy="beam_search",
        )

        greedy = translator.translate(
            source_ids,
            max_new_tokens=4,
            decoding_strategy="greedy",
        )
        torch.manual_seed(123)
        sampled = translator.translate(
            source_ids,
            max_new_tokens=4,
            decoding_strategy="nucleus_sampling",
            top_p=0.9,
        )
        beam = translator.translate(
            source_ids,
            max_new_tokens=4,
            decoding_strategy="beam_search",
            num_beams=2,
        )

        self.assertEqual(len(greedy), 1)
        self.assertEqual(len(sampled), 1)
        self.assertEqual(len(beam), 1)
        self.assertEqual(translator.beam_size, 3)
        self.assertEqual(translator.decoding_strategy, "beam_search")

    def test_encoder_decoder_finished_beam_keeps_score_and_adds_pad(self):
        translator = EncoderDecoderTranslator(
            self.build_encoder_decoder(),
            beam_size=2,
            max_seq_len=8,
        )
        generated = torch.tensor(
            [
                [1, 2, 0, 0],
                [1, 4, 0, 0],
            ]
        )
        logits = torch.zeros(2, 1, 8)
        logits[0, 0, 7] = 100.0
        scores = torch.tensor([[-0.1, -0.2]])
        finished = torch.tensor([True, False])
        config = EncoderDecoderGenerationConfig(
            strategy="beam_search",
            max_new_tokens=3,
            beam_size=2,
            temperature=0.8,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
            use_kv_cache=True,
        )

        next_generated, next_scores, next_finished = (
            translator._select_beams(
                generated,
                logits,
                scores,
                finished,
                step=2,
                config=config,
            )
        )

        finished_rows = next_generated[:, 1] == translator.trg_eos_idx
        self.assertTrue(finished_rows.any())
        self.assertTrue(
            torch.all(
                next_generated[finished_rows, 2]
                == translator.trg_pad_idx
            )
        )
        torch.testing.assert_close(
            next_scores.reshape(-1)[finished_rows],
            torch.tensor([-0.1]),
        )
        self.assertTrue(next_finished[finished_rows].all())

    def test_encoder_decoder_rejects_source_beyond_model_limit(self):
        translator = EncoderDecoderTranslator(
            self.build_encoder_decoder(),
            beam_size=2,
            max_seq_len=8,
        )

        with self.assertRaisesRegex(ValueError, "max_src_len"):
            translator.translate(torch.ones(1, 13, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
