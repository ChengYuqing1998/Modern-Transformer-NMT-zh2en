import tempfile
import unittest
from pathlib import Path

from datasets.corpus import prepare_parallel_corpus
from datasets.decoder_only import make_data, prepare_data
from datasets.encoder_decoder import prepare_data as prepare_encoder_decoder


class DatasetPipelineTest(unittest.TestCase):
    def test_language_codes_are_zh_and_en(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "zh-en.txt"
            corpus.write_text("你好\thello world\n", encoding="utf-8")

            source, target, _ = prepare_encoder_decoder(
                "zh",
                "en",
                max_target_sentence_split_length=16,
                data_path=corpus,
            )

            self.assertEqual(source.name, "zh")
            self.assertEqual(target.name, "en")

    def test_reverse_direction_uses_word_source_and_character_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "zh-en.txt"
            corpus.write_text("你好\thello world\n", encoding="utf-8")
            tokenizer, pairs, source_language, target_language = prepare_data(
                "zh",
                "en",
                max_target_sentence_split_length=16,
                reverse=True,
                data_path=corpus,
            )

            inputs, _, source_lengths, sequence_length = make_data(
                tokenizer,
                pairs,
                max_context_len=8,
                src_lang_name=source_language,
                trg_lang_name=target_language,
            )

            self.assertEqual(source_language, "en")
            self.assertEqual(target_language, "zh")
            self.assertEqual(source_lengths.tolist(), [2])
            self.assertEqual(sequence_length, 6)
            self.assertEqual(inputs[0, 1:3].tolist(), tokenizer.encode("hello world"))

    def test_target_sentence_split_length_is_inclusive_upper_bound(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "zh-en.txt"
            corpus.write_text(
                "短句\tone two\n"
                "长句\tone two three\n",
                encoding="utf-8",
            )

            pairs, _, _ = prepare_parallel_corpus(
                "zh",
                "en",
                data_path=corpus,
                max_target_length=3,
            )

            self.assertEqual(
                pairs,
                [["短句", "one two"], ["长句", "one two three"]],
            )

    def test_decoder_only_sequence_length_is_checked_against_context_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "zh-en.txt"
            corpus.write_text("你好\thello world\n", encoding="utf-8")
            tokenizer, pairs, source_language, target_language = prepare_data(
                "zh",
                "en",
                max_target_sentence_split_length=16,
                data_path=corpus,
            )

            inputs, targets, source_lengths, sequence_length = make_data(
                tokenizer,
                pairs,
                max_context_len=8,
                src_lang_name=source_language,
                trg_lang_name=target_language,
            )

            self.assertEqual(sequence_length, 6)
            self.assertEqual(inputs.shape[1], 6)
            self.assertEqual(targets.shape[1], 6)
            self.assertEqual(source_lengths.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
