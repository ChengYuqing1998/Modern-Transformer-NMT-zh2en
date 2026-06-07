import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from models.transformer import build_GPT, build_model
from scripts.inference import load_runtime
from tokenizer.tokenizer import Vocabulary
from trainer.trainer import BaseTrainer


class CheckpointWriter(BaseTrainer):
    def __init__(self, architecture, model, config):
        self.architecture = architecture
        self.model = model
        self.write_config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = None
        self.global_step = 7


class CheckpointRoundTripTest(unittest.TestCase):
    def save_tokenizer(self, tokenizer, path):
        with Path(path).open("wb") as file:
            pickle.dump(tokenizer, file)

    def test_encoder_decoder_checkpoint_rebuilds_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Vocabulary("zh")
            source.add_text("你好", character_level=True)
            target = Vocabulary("en")
            target.add_text("hello world")
            source_path = Path(temp_dir) / "source.pkl"
            target_path = Path(temp_dir) / "target.pkl"
            self.save_tokenizer(source, source_path)
            self.save_tokenizer(target, target_path)
            config = {
                "model_dim": 16,
                "hidden_dim": 32,
                "n_layer": 1,
                "n_head": 4,
                "drop_prob": 0.0,
                "pad_token": 0,
                "bos_token": 1,
                "eos_token": 2,
                "beam_size": 2,
                "inference_decoding_strategy": "nucleus_sampling",
                "inference_temperature": 0.7,
                "inference_top_p": 0.85,
                "inference_top_k": 4,
                "inference_repetition_penalty": 1.1,
                "inference_max_new_tokens": 9,
                "_device": "cpu",
                "tokenizer_paths": {
                    "source": str(source_path),
                    "target": str(target_path),
                },
            }
            model = build_model(source, target, 12, 12, config)
            writer = CheckpointWriter("encoder_decoder", model, config)
            checkpoint_path = writer._save(temp_dir, "encoder.pt", 3)
            self.assertTrue(
                (Path(checkpoint_path) / "model.safetensors").is_file()
            )

            architecture, translator, _, _ = load_runtime(
                checkpoint_path, "auto", torch.device("cpu")
            )

            self.assertEqual(architecture, "encoder_decoder")
            self.assertEqual(translator.model.max_src_len, 12)
            self.assertEqual(translator.model.max_trg_len, 12)
            self.assertEqual(
                translator.decoding_strategy, "nucleus_sampling"
            )
            self.assertEqual(translator.temperature, 0.7)
            self.assertEqual(translator.top_p, 0.85)
            self.assertEqual(translator.top_k, 4)
            self.assertEqual(translator.repetition_penalty, 1.1)
            self.assertEqual(translator.inference_max_new_tokens, 9)

    def test_decoder_only_checkpoint_rebuilds_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer = Vocabulary("mixed")
            tokenizer.add_text("你好", character_level=True)
            tokenizer.add_text("hello world")
            tokenizer_path = Path(temp_dir) / "unified.pkl"
            self.save_tokenizer(tokenizer, tokenizer_path)
            config = {
                "model_dim": 16,
                "hidden_dim": 32,
                "n_layer": 1,
                "n_head": 4,
                "drop_prob": 0.0,
                "pad_token": 0,
                "bos_token": 1,
                "eos_token": 2,
                "use_gqa": False,
                "use_attention_sink": False,
                "use_rope": True,
                "use_swiglu": True,
                "use_rms_norm": True,
                "use_pre_norm": True,
                "inference_max_new_tokens": 10,
                "_device": "cpu",
                "tokenizer_paths": {"unified": str(tokenizer_path)},
            }
            model = build_GPT(tokenizer, 12, config)
            writer = CheckpointWriter("decoder_only", model, config)
            checkpoint_path = writer._save(temp_dir, "decoder.pt", 3)

            architecture, translator, _, _ = load_runtime(
                checkpoint_path, "auto", torch.device("cpu")
            )

            self.assertEqual(architecture, "decoder_only")
            self.assertEqual(translator.model.max_context_len, 12)
            self.assertEqual(translator.inference_max_new_tokens, 10)


if __name__ == "__main__":
    unittest.main()
