import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from models.transformer import build_GPT, build_model
from tokenizer.tokenizer import Vocabulary
from trainer.trainer import (
    BaseTrainer,
    DecoderOnlyTrainer,
    EncoderDecoderTrainer,
    _resolve_trial_paths,
)


class PolicyCheckpointWriter(BaseTrainer):
    architecture = "encoder_decoder"

    def __init__(self, directory):
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = None
        self.global_step = 0
        self.best_loss = float("inf")
        self.device = torch.device("cpu")
        self.ckpt_dir = directory
        self.ckpt_file_name = "model.pt"
        self.write_config = {
            "save_total_limit": 2,
            "tokenizer_paths": {},
        }
        self._configure_training_policy()


class TrainingPolicyTest(unittest.TestCase):
    def test_trial_name_derives_checkpoint_directory_and_log_name(self):
        trial_name, checkpoint_dir, log_file_name = _resolve_trial_paths(
            {
                "trial_name": "c2e-gpt-ablation",
                "ckpt_dir": "./checkpoints/decoder_only",
            }
        )

        self.assertEqual(trial_name, "c2e-gpt-ablation")
        self.assertEqual(
            checkpoint_dir,
            "checkpoints/decoder_only/c2e-gpt-ablation",
        )
        self.assertEqual(log_file_name, "c2e-gpt-ablation.log")

    def test_checkpoint_rotation_counts_best_toward_total_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            for step in range(1, 4):
                writer.global_step = step
                writer._save_periodic_checkpoint(epoch=1)
            writer._save_best_checkpoint(1, {"val_loss": 0.5})

            checkpoints = [
                path
                for path in Path(temp_dir).glob("model-*")
                if path.is_dir()
            ]
            periodic = [
                path
                for path in checkpoints
                if not path.name.endswith("-best")
            ]
            best = (
                Path(temp_dir)
                / "model-epoch-0001-step-00000003-best"
            )

            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(len(periodic), 1)
            self.assertTrue(best.is_dir())
            self.assertTrue((best / "model.safetensors").is_file())
            self.assertTrue((best / "metadata.json").is_file())
            self.assertTrue((best / "trainer_state.pt").is_file())
            self.assertTrue((best / "optimizer.pt").is_file())

    def test_new_best_replaces_old_best_and_keeps_total_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            writer.global_step = 1
            writer._save_best_checkpoint(1, {"val_loss": 0.8})
            writer.global_step = 2
            writer._save_periodic_checkpoint(1, {"val_loss": 0.7})
            writer._save_best_checkpoint(1, {"val_loss": 0.7})
            writer.global_step = 3
            writer._save_periodic_checkpoint(1, {"val_loss": 0.9})

            checkpoints = sorted(
                path
                for path in Path(temp_dir).glob("model-*")
                if path.is_dir()
            )
            best = [
                path
                for path in checkpoints
                if path.name.endswith("-best")
            ]

            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(
                [path.name for path in best],
                ["model-epoch-0001-step-00000002-best"],
            )
            self.assertTrue(
                (best[0] / "model.safetensors").is_file()
            )

    def test_old_best_becomes_regular_and_recent_three_are_kept(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            writer.save_total_limit = 3
            for step in range(1, 5):
                writer.global_step = step
                writer._save_periodic_checkpoint(epoch=1)
                writer._save_best_checkpoint(
                    1, {"val_loss": 1.0 / step}
                )

            names = sorted(
                path.name
                for path in Path(temp_dir).glob("model-*")
                if path.is_dir()
            )

            self.assertEqual(
                names,
                [
                    "model-epoch-0001-step-00000002",
                    "model-epoch-0001-step-00000003",
                    "model-epoch-0001-step-00000004-best",
                ],
            )

    def test_old_best_plus_two_recent_when_best_is_not_recent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            writer.save_total_limit = 3
            writer.global_step = 1
            writer._save_best_checkpoint(1, {"val_loss": 0.1})
            for step in range(2, 5):
                writer.global_step = step
                writer._save_periodic_checkpoint(epoch=1)

            names = sorted(
                path.name
                for path in Path(temp_dir).glob("model-*")
                if path.is_dir()
            )

            self.assertEqual(
                names,
                [
                    "model-epoch-0001-step-00000001-best",
                    "model-epoch-0001-step-00000003",
                    "model-epoch-0001-step-00000004",
                ],
            )

    def test_global_step_counts_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_tokenizer = Vocabulary("zh")
            source_tokenizer.add_text("你好", character_level=True)
            target_tokenizer = Vocabulary("en")
            target_tokenizer.add_text("hello world")
            config = {
                "model_dim": 16,
                "hidden_dim": 32,
                "n_layer": 1,
                "n_head": 4,
                "drop_prob": 0.0,
                "pad_token": 0,
                "bos_token": 1,
                "eos_token": 2,
                "_device": "cpu",
                "gradient_accumulation_steps": 2,
                "print_every_n_steps": 10,
                "eval_strategy": "epoch",
                "eval_interval": 1,
                "save_strategy": "epoch",
                "save_interval": 1,
                "save_total_limit": 2,
            }
            model = build_model(
                source_tokenizer, target_tokenizer, 8, 8, config
            )
            trainer = EncoderDecoderTrainer(
                model=model,
                ckpt_dir=temp_dir,
                ckpt_file_name="model.pt",
                log_dir=temp_dir,
                log_file_name="train.log",
                device=torch.device("cpu"),
                write_config=config,
            )
            trainer.build_loss("ce", 0.0, 0)
            trainer.build_optimizer(1e-3, "adam")
            trainer.optimizer.zero_grad(set_to_none=True)
            source = torch.tensor([[1, 4, 5, 2]])
            target = torch.tensor([[1, 4, 5, 2]])

            trainer.fit_iter(
                source,
                target,
                accumulation_divisor=2,
                should_update=False,
            )
            self.assertEqual(trainer.global_step, 0)
            trainer.fit_iter(
                source,
                target,
                accumulation_divisor=2,
                should_update=True,
            )
            self.assertEqual(trainer.global_step, 1)

    def test_decoder_only_label_smoothing_runs_training_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer = Vocabulary("mixed")
            tokenizer.add_text("你好", character_level=True)
            tokenizer.add_text("hello world")
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
                "_device": "cpu",
                "gradient_accumulation_steps": 1,
                "print_every_n_steps": 10,
                "eval_strategy": "epoch",
                "eval_interval": 1,
                "save_strategy": "epoch",
                "save_interval": 1,
                "save_total_limit": 2,
            }
            model = build_GPT(tokenizer, 8, config)
            trainer = DecoderOnlyTrainer(
                model=model,
                ckpt_dir=temp_dir,
                ckpt_file_name="model.pt",
                log_dir=temp_dir,
                log_file_name="decoder.log",
                device=torch.device("cpu"),
                write_config=config,
            )
            trainer.build_loss("kl", smoothing=0.1, ignore_index=0)
            trainer.build_optimizer(1e-3, "adam")
            trainer.optimizer.zero_grad(set_to_none=True)

            with patch("trainer.trainer.wandb.log"):
                loss = trainer.fit_iter(
                    torch.tensor([[1, 4, 5, 6]]),
                    torch.tensor([[4, 5, 6, 2]]),
                    torch.tensor([2]),
                )

            self.assertGreater(loss, 0)
            self.assertEqual(trainer.global_step, 1)

    def test_bf16_autocast_runs_forward_and_backward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_tokenizer = Vocabulary("zh")
            source_tokenizer.add_text("你好", character_level=True)
            target_tokenizer = Vocabulary("en")
            target_tokenizer.add_text("hello world")
            config = {
                "model_dim": 16,
                "hidden_dim": 32,
                "n_layer": 1,
                "n_head": 4,
                "drop_prob": 0.0,
                "pad_token": 0,
                "bos_token": 1,
                "eos_token": 2,
                "_device": "cpu",
                "train_precision": "bf16_mixed",
                "checkpoint_precision": "bf16",
                "gradient_accumulation_steps": 1,
                "print_every_n_steps": 10,
                "eval_strategy": "epoch",
                "eval_interval": 1,
                "save_strategy": "epoch",
                "save_interval": 1,
                "save_total_limit": 2,
            }
            model = build_model(
                source_tokenizer, target_tokenizer, 8, 8, config
            )
            trainer = EncoderDecoderTrainer(
                model=model,
                ckpt_dir=temp_dir,
                ckpt_file_name="model.pt",
                log_dir=temp_dir,
                log_file_name="bf16.log",
                device=torch.device("cpu"),
                write_config=config,
            )
            trainer.build_loss("ce", 0.0, 0)
            trainer.build_optimizer(1e-3, "adam")
            trainer.optimizer.zero_grad(set_to_none=True)

            loss = trainer.fit_iter(
                torch.tensor([[1, 4, 5, 2]]),
                torch.tensor([[1, 4, 5, 2]]),
            )

            self.assertGreater(loss, 0)
            self.assertEqual(trainer.global_step, 1)

    def test_fp32_full_disables_autocast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_tokenizer = Vocabulary("zh")
            source_tokenizer.add_text("你好", character_level=True)
            target_tokenizer = Vocabulary("en")
            target_tokenizer.add_text("hello world")
            config = {
                "model_dim": 16,
                "hidden_dim": 32,
                "n_layer": 1,
                "n_head": 4,
                "drop_prob": 0.0,
                "pad_token": 0,
                "bos_token": 1,
                "eos_token": 2,
                "_device": "cpu",
                "train_precision": "fp32_full",
                "checkpoint_precision": "bf16",
                "gradient_accumulation_steps": 1,
                "print_every_n_steps": 10,
                "eval_strategy": "epoch",
                "eval_interval": 1,
                "save_strategy": "epoch",
                "save_interval": 1,
                "save_total_limit": 2,
            }
            model = build_model(
                source_tokenizer, target_tokenizer, 8, 8, config
            )
            trainer = EncoderDecoderTrainer(
                model=model,
                ckpt_dir=temp_dir,
                ckpt_file_name="model.pt",
                log_dir=temp_dir,
                log_file_name="fp32.log",
                device=torch.device("cpu"),
                write_config=config,
            )
            self.assertEqual(
                trainer._autocast_context().__class__.__name__,
                "nullcontext",
            )

    def test_checkpoint_model_weights_are_saved_as_bf16(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            writer.model.to(dtype=torch.float32)
            checkpoint_path = writer._save(temp_dir, "bf16.pt", epoch=1)
            state_dict = load_file(
                str(Path(checkpoint_path) / "model.safetensors")
            )

            floating_dtypes = {
                tensor.dtype
                for tensor in state_dict.values()
                if tensor.is_floating_point()
            }

            self.assertEqual(floating_dtypes, {torch.bfloat16})

    def test_checkpoint_precision_can_remain_fp32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            writer.checkpoint_precision = "fp32"
            checkpoint_path = writer._save(temp_dir, "fp32.pt", epoch=1)
            state_dict = load_file(
                str(Path(checkpoint_path) / "model.safetensors")
            )

            floating_dtypes = {
                tensor.dtype
                for tensor in state_dict.values()
                if tensor.is_floating_point()
            }

            self.assertEqual(floating_dtypes, {torch.float32})

    def test_resume_restores_training_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            inputs = torch.ones(1, 2)
            writer.optimizer.zero_grad()
            writer.model(inputs).sum().backward()
            writer.optimizer.step()
            writer.global_step = 12
            writer.best_loss = 0.25
            checkpoint_path = writer._save(temp_dir, "resume.pt", epoch=4)
            self.assertTrue(Path(checkpoint_path).is_dir())

            restored = PolicyCheckpointWriter(temp_dir)
            restored.resume_from_checkpoint(checkpoint_path)

            self.assertEqual(restored.global_step, 12)
            self.assertEqual(restored.start_epoch, 5)
            self.assertEqual(restored.best_loss, 0.25)
            self.assertTrue(restored.optimizer.state)

    def test_step_checkpoint_resumes_inside_saved_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = PolicyCheckpointWriter(temp_dir)
            writer.global_step = 9
            writer.micro_batch_in_epoch = 18
            writer.epoch_complete = False
            checkpoint_path = writer._save(temp_dir, "step.pt", epoch=3)

            restored = PolicyCheckpointWriter(temp_dir)
            restored.resume_from_checkpoint(checkpoint_path)

            self.assertEqual(restored.start_epoch, 3)
            self.assertEqual(restored.resume_micro_batch, 18)
            self.assertEqual(restored.global_step, 9)

    def test_event_strategy_uses_requested_unit(self):
        self.assertTrue(
            BaseTrainer._event_due("step", 5, 3, 10, "step")
        )
        self.assertFalse(
            BaseTrainer._event_due("step", 5, 10, 3, "epoch")
        )
        self.assertTrue(
            BaseTrainer._event_due("epoch", 2, 4, 99, "epoch")
        )


if __name__ == "__main__":
    unittest.main()
