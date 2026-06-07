import pickle
import json
import tqdm
import wandb
import os
import shutil
from pathlib import Path
from torch.optim import lr_scheduler
import logging
from contextlib import nullcontext
import random
import numpy as np
from tqdm import tqdm
from torch.nn.functional import log_softmax
import torch
import torch.nn as nn
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from trainer.checkpoint import (
    load_checkpoint_metadata,
    load_checkpoint_model,
    load_checkpoint_training_state,
    save_checkpoint,
)
from inference.translator import (
    DecoderOnlyTranslator,
    EncoderDecoderTranslator,
    GPTGenerator,
)

try:
    from experiments.muon import Muon, separate_muon_params
except ImportError:
    Muon = None
    separate_muon_params = None


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def build_decoder_only_trainer(model, configs):
    trainer = DecoderOnlyTrainer(model=model,
                        ckpt_dir=configs['ckpt_dir'],
                        ckpt_file_name=configs['ckpt_file_name'],
                        log_dir=configs['log_dir'],
                        log_file_name=configs['log_file_name'],
                        device=eval(configs['device']),
                        write_config=configs
                        )
    trainer.build_loss(configs['loss_type'], configs['smoothing'], configs['ignore_index'])

    # Build optimizer with Muon-specific parameters if needed
    optimizer_kwargs = {}
    if configs['optimizer_type'] == 'muon':
        optimizer_kwargs = {
            'muon_lr': configs.get('muon_lr', 0.02),
            'muon_momentum': configs.get('muon_momentum', 0.95),
            'muon_nesterov': configs.get('muon_nesterov', True),
            'muon_ns_steps': configs.get('muon_ns_steps', 5),
            'muon_adamw_lr': configs.get('learning_rate', 3e-4),
            'muon_adamw_betas': configs.get('muon_adamw_betas', (0.9, 0.95)),
            'muon_adamw_eps': configs.get('muon_adamw_eps', 1e-8),
            'muon_adamw_wd': configs.get('muon_adamw_wd', 0.0),
        }

    trainer.build_optimizer(configs['learning_rate'], configs['optimizer_type'], **optimizer_kwargs)

    if configs['scheduler_flag']:
        trainer.build_scheduler(configs['anneal_rate'],
                               configs['scheduler_type'],
                               configs['patience'],
                               configs['threshold'])
    return trainer


build_trainer_gpt = build_decoder_only_trainer


def remove_element(lst, element):
    if isinstance(lst, list):
        return [remove_element(sublst, element) for sublst in lst if sublst != element]
    else:
        return lst if lst != element else None


class BaseTrainer:
    architecture = None

    def _configure_training_policy(self):
        config = self.write_config
        self.gradient_accumulation_steps = int(
            config.get("gradient_accumulation_steps", 1)
        )
        self.print_every_n_steps = int(
            config.get("print_every_n_steps", 100)
        )
        self.eval_strategy = config.get("eval_strategy", "epoch")
        self.eval_interval = int(config.get("eval_interval", 1))
        self.save_strategy = config.get("save_strategy", "epoch")
        self.save_interval = int(config.get("save_interval", 1))
        self.save_total_limit = int(config.get("save_total_limit", 3))
        self.save_best = bool(config.get("save_best", True))
        self.show_eval_sample = bool(config.get("show_eval_sample", False))
        self.eval_sample_sentence = config.get("eval_sample_sentence", "")
        train_precision = config.get(
            "train_precision", config.get("precision", "fp32_full")
        )
        precision_aliases = {
            "fp32": "fp32_full",
            "bf16": "bf16_mixed",
        }
        self.train_precision = precision_aliases.get(
            train_precision, train_precision
        )
        self.checkpoint_precision = config.get(
            "checkpoint_precision", "bf16"
        )
        self.start_epoch = 1
        self.resume_micro_batch = 0
        self.micro_batch_in_epoch = 0
        self.epoch_complete = True
        for name, value in (
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("print_every_n_steps", self.print_every_n_steps),
            ("eval_interval", self.eval_interval),
            ("save_interval", self.save_interval),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("eval_strategy", self.eval_strategy),
            ("save_strategy", self.save_strategy),
        ):
            if value not in ("step", "epoch"):
                raise ValueError(f"{name} must be 'step' or 'epoch'")
        if self.save_total_limit < 1:
            raise ValueError("save_total_limit must be at least 1")
        if self.train_precision not in ("fp32_full", "bf16_mixed"):
            raise ValueError(
                "train_precision must be 'fp32_full' or 'bf16_mixed'"
            )
        if self.checkpoint_precision not in ("fp32", "bf16"):
            raise ValueError(
                "checkpoint_precision must be 'fp32' or 'bf16'"
            )
        if (
            self.train_precision == "bf16_mixed"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("The selected CUDA device does not support BF16")

    def _autocast_context(self):
        if self.train_precision != "bf16_mixed":
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

    def _checkpoint_metadata(self, epoch, metrics=None, is_best=False):
        metadata = {
            "format_version": 2,
            "checkpoint_format": "safetensors",
            "architecture": self.architecture,
            "epoch": epoch,
            "global_step": self.global_step,
            "model_config": dict(self.write_config),
            "tokenizer_paths": dict(
                self.write_config.get("tokenizer_paths", {})
            ),
            "checkpoint_precision": getattr(
                self, "checkpoint_precision", "bf16"
            ),
            "metrics": metrics or {},
            "best_eval_loss": getattr(self, "best_loss", float("inf")),
            "is_best": bool(is_best),
        }
        if self.architecture == "encoder_decoder":
            metadata["max_src_len"] = getattr(self.model, "max_src_len", None)
            metadata["max_trg_len"] = getattr(self.model, "max_trg_len", None)
        else:
            metadata["max_context_len"] = getattr(
                self.model, "max_context_len", None
            )
        return metadata

    def _training_state(self, epoch):
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_eval_loss": getattr(self, "best_loss", float("inf")),
            "micro_batch_in_epoch": getattr(
                self, "micro_batch_in_epoch", 0
            ),
            "epoch_complete": getattr(self, "epoch_complete", True),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    def _save(
        self,
        ckpt_dir,
        ckpt_file_name,
        epoch,
        metrics=None,
        is_best=False,
    ):
        checkpoint_dir = Path(ckpt_dir) / Path(ckpt_file_name).stem
        return save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=self.model,
            metadata=self._checkpoint_metadata(
                epoch, metrics, is_best=is_best
            ),
            trainer_state=self._training_state(epoch),
            optimizer_state=(
                self.optimizer.state_dict()
                if self.optimizer is not None
                else None
            ),
            scheduler_state=(
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            is_best=is_best,
        )

    def _checkpoint_name(self, epoch, is_best=False):
        configured = Path(self.ckpt_file_name)
        name = (
            f"{configured.stem}-epoch-{epoch:04d}"
            f"-step-{self.global_step:08d}"
        )
        return f"{name}-best" if is_best else name

    def _save_periodic_checkpoint(self, epoch, metrics=None):
        path = self._save(
            self.ckpt_dir,
            self._checkpoint_name(epoch),
            epoch,
            metrics,
        )
        self._rotate_checkpoints()
        return path

    def _save_best_checkpoint(self, epoch, metrics):
        checkpoint_root = Path(self.ckpt_dir)
        best_name = self._checkpoint_name(epoch, is_best=True)
        path = self._save(
            self.ckpt_dir,
            best_name,
            epoch,
            metrics,
            is_best=True,
        )
        normal_path = checkpoint_root / self._checkpoint_name(epoch)
        if normal_path.is_dir():
            shutil.rmtree(normal_path)

        configured = Path(self.ckpt_file_name)
        for previous_best in checkpoint_root.glob(
            f"{configured.stem}-epoch-*-step-*-best"
        ):
            if previous_best.is_dir() and previous_best.name != best_name:
                regular_path = previous_best.with_name(
                    previous_best.name[:-len("-best")]
                )
                if regular_path.exists():
                    shutil.rmtree(regular_path)
                previous_best.rename(regular_path)
                self._set_checkpoint_best_status(
                    regular_path, is_best=False
                )
        self._rotate_checkpoints()
        return path

    @staticmethod
    def _set_checkpoint_best_status(checkpoint_path, is_best):
        metadata_path = Path(checkpoint_path) / "metadata.json"
        if not metadata_path.is_file():
            return
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["is_best"] = bool(is_best)
        metadata["model_file"] = "model.safetensors"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    def _rotate_checkpoints(self):
        configured = Path(self.ckpt_file_name)
        pattern = f"{configured.stem}-epoch-*-step-*"
        checkpoints = list(
            (
                path
                for path in Path(self.ckpt_dir).glob(pattern)
                if path.is_dir()
            )
        )
        checkpoints.sort(key=self._checkpoint_sort_key, reverse=True)
        recent = checkpoints[:self.save_total_limit]
        best_path = next(
            (
                path
                for path in checkpoints
                if path.name.endswith("-best")
            ),
            None,
        )
        if best_path is None or best_path in recent:
            keep = set(recent)
        else:
            keep = {best_path}
            keep.update(recent[:self.save_total_limit - 1])
        for stale_path in checkpoints:
            if stale_path not in keep:
                shutil.rmtree(stale_path)

    @staticmethod
    def _checkpoint_sort_key(path):
        name = (
            path.name[:-len("-best")]
            if path.name.endswith("-best")
            else path.name
        )
        try:
            epoch_text, step_text = name.rsplit("-epoch-", 1)[1].split(
                "-step-", 1
            )
            return int(epoch_text), int(step_text)
        except (IndexError, ValueError):
            return -1, -1

    @staticmethod
    def _event_due(strategy, interval, epoch, global_step, event):
        value = global_step if strategy == "step" else epoch
        return event == strategy and value > 0 and value % interval == 0

    def _optimizer_update(self, clip=None):
        if clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def resume_from_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            metadata = load_checkpoint_metadata(checkpoint_path)
            checkpoint = load_checkpoint_training_state(
                checkpoint_path, self.device
            )
            checkpoint.update(metadata)
            load_checkpoint_model(
                self.model, checkpoint_path, self.device
            )
        else:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        if checkpoint.get("architecture") != self.architecture:
            raise ValueError(
                "Checkpoint architecture does not match trainer architecture"
            )
        if not checkpoint_path.is_dir():
            self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_loss = float(
            checkpoint.get("best_eval_loss", float("inf"))
        )
        saved_epoch = int(checkpoint.get("epoch", 0))
        if checkpoint.get("epoch_complete", True):
            self.start_epoch = saved_epoch + 1
            self.resume_micro_batch = 0
        else:
            self.start_epoch = saved_epoch
            self.resume_micro_batch = int(
                checkpoint.get("micro_batch_in_epoch", 0)
            )
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(
            f"Resumed {self.architecture} training from epoch "
            f"{self.start_epoch}, micro-batch {self.resume_micro_batch}, "
            f"step {self.global_step}"
        )
        return checkpoint

    def _step_scheduler_after_eval(self, eval_loss):
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(eval_loss)

    def _step_scheduler_after_epoch(self, epoch, warmup):
        if (
            self.scheduler is not None
            and not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)
            and epoch > warmup
        ):
            self.scheduler.step()

    @staticmethod
    def make_log(log_dir, log_file_name):
        path = os.path.join(log_dir, log_file_name)
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(path)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(path)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(filename)s[line:%(lineno)d] "
                    "%(levelname)s %(message)s"
                )
            )
            logger.addHandler(handler)
        return logger


class DecoderOnlyTrainer(BaseTrainer):
    architecture = "decoder_only"

    def __init__(self, model: nn.Module, ckpt_dir, ckpt_file_name, log_dir,
                 log_file_name, device, write_config, **kwargs):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.ckpt_file_name = ckpt_file_name
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.device = device
        self.write_config = write_config

        self.optimizer = None
        self.loss = None
        self.scheduler = None
        self.global_step = 0
        self.best_loss = 1e9
        self._configure_training_policy()

        # GPT Generator for inference (low-level)
        self.generator = GPTGenerator(
            model,
            pad_token_id=write_config['pad_token'],
            eos_token_id=write_config['eos_token'],
            bos_token_id=write_config['bos_token']
        )

        # Translator will be initialized lazily when needed
        self.translator = None

        self.log = self.make_log(log_dir, log_file_name)
        self.log.info(msg=self.write_config)

    def build_optimizer(self, learning_rate, optimizer_type, **kwargs):
        assert optimizer_type in ['sgd', 'adam', 'muon']
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'muon':
            if Muon is None:
                raise ImportError("optimizer_type='muon' requires the optional muon.py module")
            # Muon optimizer with parameter separation
            param_groups = separate_muon_params(self.model)
            self.optimizer = Muon(
                param_groups,
                lr=kwargs.get('muon_lr', 0.02),
                momentum=kwargs.get('muon_momentum', 0.95),
                nesterov=kwargs.get('muon_nesterov', True),
                ns_steps=kwargs.get('muon_ns_steps', 5),
                adamw_lr=kwargs.get('muon_adamw_lr', learning_rate),  # Use config lr for adamw part
                adamw_betas=kwargs.get('muon_adamw_betas', (0.9, 0.95)),
                adamw_eps=kwargs.get('muon_adamw_eps', 1e-8),
                adamw_wd=kwargs.get('muon_adamw_wd', 0.0),
            )
            print(f"Using Muon optimizer:")
            print(f"  Muon group LR: {kwargs.get('muon_lr', 0.02)}")
            print(f"  AdamW group LR: {learning_rate}")
            print(f"  Momentum: {kwargs.get('muon_momentum', 0.95)}")

    def build_scheduler(self, anneal_rate, scheduler_type, patience, threshold):
        assert scheduler_type in ['exp', 'plateau', 'cosine']
        if scheduler_type == 'exp':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, anneal_rate)
        elif scheduler_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode='min',
                                                            patience=patience,
                                                            factor=anneal_rate,
                                                            threshold=threshold,
                                                            threshold_mode='abs')
        elif scheduler_type == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                            T_max=self.write_config['max_epochs'],
                                                            eta_min=self.write_config['learning_rate'] * 1e-2)

    def get_translator(self, unified_lang):
        """
        Get or create translator (lazy initialization to avoid creating it if not needed)
        """
        if self.translator is None:
            self.translator = DecoderOnlyTranslator(
                self.model,
                unified_lang,
                device=self.device,
                use_kv_cache=self.write_config.get(
                    "inference_use_kv_cache", True
                ),
                decoding_strategy=self.write_config.get(
                    "inference_decoding_strategy", "beam_search"
                ),
                num_beams=self.write_config.get("beam_size", 5),
                temperature=self.write_config.get(
                    "inference_temperature", 0.8
                ),
                top_p=self.write_config.get("inference_top_p", 0.9),
                top_k=self.write_config.get("inference_top_k", 0),
                repetition_penalty=self.write_config.get(
                    "inference_repetition_penalty", 1.0
                ),
                inference_max_new_tokens=self.write_config.get(
                    "inference_max_new_tokens", 160
                ),
            )
        return self.translator

    def build_loss(self, loss_type, smoothing, ignore_index=None, **kwargs):
        assert loss_type in ['ce', 'nll', 'kl']
        # For GPT, use unified vocab_size
        vocab_size = getattr(self.model, 'vocab_size', None)
        if vocab_size is None:
            # Try to get from generator if available
            if hasattr(self.model, 'generator'):
                vocab_size = self.model.generator.proj.out_features
            else:
                raise ValueError("Cannot determine vocab_size for GPT model")

        if loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=int(ignore_index))
        elif loss_type == 'nll':
            self.loss = torch.nn.NLLLoss(ignore_index=int(ignore_index))
        elif loss_type == 'kl':
            self.loss = LabelSmoothing(vocab_size, int(ignore_index), smoothing)

    def fit(self, train_data, val_data, unified_lang, max_epochs, warmup,
            test_data=None, clip=None, **kwargs):
        """
        Train GPT model
        train_data: DataLoader that returns (inputs, targets, src_lengths)
        val_data: validation DataLoader
        test_data: test DataLoader (optional, for monitoring test BLEU during training)
        unified_lang: unified language object for both source and target
        """
        del test_data, kwargs
        for epoch in tqdm(range(self.start_epoch, max_epochs + 1)):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_loss_in_epoch = []
            if getattr(train_data, "generator", None) is not None:
                train_data.generator.manual_seed(
                    int(self.write_config.get("random_seed", 0)) + epoch
                )
            num_micro_batches = len(train_data)
            for micro_batch, (inputs, targets, src_lengths) in enumerate(
                train_data, start=1
            ):
                if (
                    epoch == self.start_epoch
                    and micro_batch <= self.resume_micro_batch
                ):
                    continue
                self.micro_batch_in_epoch = micro_batch
                self.epoch_complete = False
                group_start = (
                    (micro_batch - 1) // self.gradient_accumulation_steps
                ) * self.gradient_accumulation_steps + 1
                accumulation_divisor = min(
                    self.gradient_accumulation_steps,
                    num_micro_batches - group_start + 1,
                )
                should_update = (
                    micro_batch % self.gradient_accumulation_steps == 0
                    or micro_batch == num_micro_batches
                )
                loss = self.fit_iter(
                    inputs,
                    targets,
                    src_lengths,
                    accumulation_divisor=accumulation_divisor,
                    should_update=should_update,
                    clip=clip,
                )
                train_loss_in_epoch.append(loss)
                self.log.info(
                    "Epoch %04d micro-batch %d training loss %.6f",
                    epoch,
                    micro_batch,
                    loss,
                )
                if not should_update:
                    continue

                avg_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch)
                wandb.log(
                    {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "train_loss": loss,
                        "average_train_loss": avg_loss,
                    }
                )
                if self.global_step % self.print_every_n_steps == 0:
                    avg_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch)
                    print(
                        f"Epoch {epoch:04d} | step {self.global_step:08d} | "
                        f"average training loss {avg_loss:.6f}"
                    )

                if self._event_due(
                    self.save_strategy,
                    self.save_interval,
                    epoch,
                    self.global_step,
                    "step",
                ):
                    self._save_periodic_checkpoint(epoch)

                if self._event_due(
                    self.eval_strategy,
                    self.eval_interval,
                    epoch,
                    self.global_step,
                    "step",
                ):
                    self._evaluate_and_record(
                        val_data, unified_lang, epoch
                    )
                    self.model.train()

            if epoch == 1:
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print("total_para_counts: ", total_params)
                wandb.log({'total_para_counts': total_params})

            if self._event_due(
                self.save_strategy,
                self.save_interval,
                epoch,
                self.global_step,
                "epoch",
            ):
                self.epoch_complete = True
                self._save_periodic_checkpoint(epoch)
            if self._event_due(
                self.eval_strategy,
                self.eval_interval,
                epoch,
                self.global_step,
                "epoch",
            ):
                self.epoch_complete = True
                self._evaluate_and_record(val_data, unified_lang, epoch)
            self._step_scheduler_after_epoch(epoch, warmup)
            self.resume_micro_batch = 0
            self.epoch_complete = True

    def _evaluate_and_record(self, val_data, unified_lang, epoch):
        requested_eval_tokens = int(
            self.write_config.get("inference_max_new_tokens", 160)
        )
        inference_max_new_tokens = min(
            requested_eval_tokens,
            int(
                getattr(
                    self.model,
                    "max_trg_len",
                    getattr(self.model, "max_context_len", 160),
                )
            ),
        )
        eval_loss, eval_bleu4 = self.eval(
            val_data,
            unified_lang,
            max_new_tokens=inference_max_new_tokens,
            num_beams=self.write_config.get("beam_size", 5),
        )
        print(
            f"Epoch {epoch:04d} | step {self.global_step:08d} | "
            f"val loss {eval_loss:.6f} | BLEU-4 {eval_bleu4:.6f}"
        )
        self.log.info(
            "Epoch %04d step %08d val loss %.6f BLEU-4 %.6f",
            epoch,
            self.global_step,
            eval_loss,
            eval_bleu4,
        )
        wandb.log(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "val_loss": eval_loss,
                "val_bleu4": eval_bleu4,
            }
        )
        if self.save_best and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self._save_best_checkpoint(
                epoch,
                {"val_loss": eval_loss, "val_bleu4": eval_bleu4},
            )
        self._step_scheduler_after_eval(eval_loss)
        return eval_loss, eval_bleu4

    def fit_iter(self, inputs, targets, src_lengths,
                 accumulation_divisor=1, should_update=True, clip=None):
        """
        Single training iteration for GPT
        inputs: [batch_size, seq_len] - [BOS, src, trg, EOS, PAD, ...]
        targets: [batch_size, seq_len] - [src, trg, EOS, PAD, ...] (shifted by 1)
        src_lengths: [batch_size] - length of src part (excluding BOS)
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        src_lengths = src_lengths.to(self.device)

        # GPT forward: only needs inputs
        with self._autocast_context():
            logits = self.model.forward(inputs)

        # Apply log_softmax for NLL loss, or use raw logits for CE loss
        if isinstance(self.loss, torch.nn.NLLLoss):
            logits = log_softmax(logits, dim=-1)

        # Reshape for loss calculation: [batch_size * seq_len, vocab_size]
        logits_flat = logits.contiguous().view(-1, logits.shape[-1])
        targets_flat = targets.contiguous().view(-1).clone()

        # Vectorized masking: mask src part in targets
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        ignore_index = self.loss.ignore_index

        # Create position indices [0, 1, 2, ..., seq_len-1]
        pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]

        # Expand src_lengths to [batch_size, seq_len] for comparison
        src_lengths_expanded = src_lengths.unsqueeze(1)  # [batch_size, 1]

        # Create mask: True where position < src_length (src part), False otherwise
        src_mask = pos_indices < src_lengths_expanded  # [batch_size, seq_len]

        # Apply mask to targets_flat by reshaping and applying
        targets_reshaped = targets_flat.view(batch_size, seq_len)
        targets_reshaped = targets_reshaped.masked_fill(src_mask, ignore_index)
        targets_flat = targets_reshaped.view(-1)

        loss = self.loss(logits_flat, targets_flat)
        (loss / accumulation_divisor).backward()
        if should_update:
            self._optimizer_update(clip)
            wandb.log(
                {
                    "global_step": self.global_step,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )
        return loss.item()

    def _prepare_left_padded_src(self, inputs, src_lengths, batch_size):
        """
        Prepare left-padded source sequences for GPT inference.

        Args:
            inputs: [batch_size, seq_len] right-padded input sequences
            src_lengths: [batch_size] actual source lengths (without BOS)
            batch_size: batch size

        Returns:
            src_seqs: [batch_size, max_src_len] left-padded sequences
            attention_mask: [batch_size, max_src_len] attention mask
        """
        pad_idx = self.write_config['pad_token']
        max_src_len = src_lengths.max().item() + 1  # +1 for BOS

        # Create position indices
        positions = torch.arange(max_src_len, device=self.device).unsqueeze(0)  # [1, max_src_len]
        src_lengths_expanded = (src_lengths + 1).unsqueeze(1)  # [batch_size, 1], +1 for BOS

        # Calculate offset for left padding
        offsets = max_src_len - src_lengths_expanded  # [batch_size, 1]
        src_mask = (positions >= offsets) & (positions < max_src_len)  # [batch_size, max_src_len]

        # Calculate source positions to read from
        read_positions = positions - offsets  # [batch_size, max_src_len]
        read_positions = torch.clamp(read_positions, min=0, max=inputs.shape[1]-1)

        # Create left-padded sequences
        src_seqs = torch.full((batch_size, max_src_len), pad_idx, dtype=torch.long, device=self.device)
        src_seqs[src_mask] = inputs.gather(1, read_positions)[src_mask]

        # Create attention mask
        attention_mask = src_mask.long()

        return src_seqs, attention_mask

    def _extract_references(self, targets, src_lengths, batch_size):
        """
        Extract reference translations from targets.

        Args:
            targets: [batch_size, seq_len] target sequences
            src_lengths: [batch_size] source lengths
            batch_size: batch size

        Returns:
            refer: list of reference token lists (format for corpus_bleu)
        """
        pad_idx = self.write_config['pad_token']
        eos_idx = self.write_config['eos_token']

        seq_len = targets.shape[1]
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]
        src_lengths_expanded = src_lengths.unsqueeze(1)  # [batch_size, 1]
        ref_mask = positions >= src_lengths_expanded  # [batch_size, seq_len]

        # Apply mask and convert to list
        refer = []
        for i in range(batch_size):
            ref_tokens = targets[i][ref_mask[i]].tolist()
            # Remove padding and EOS
            ref_clean = [t for t in ref_tokens if t != pad_idx and t != eos_idx]
            refer.append([ref_clean])  # Wrap in list for corpus_bleu

        return refer

    def eval(self, val_data, unified_lang, compute_bleu=True, max_new_tokens=160, num_beams=5):
        """
        Evaluate GPT model with loss and BLEU calculation (simplified version)
        """
        self.model.eval()
        translator = self.get_translator(unified_lang)
        validate_loss = 0.0
        bleu4 = 0.0
        sample_num = 0

        with torch.no_grad():
            for inputs, targets, src_lengths in tqdm(val_data, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                src_lengths = src_lengths.to(self.device)
                batch_size = len(inputs)
                sample_num += batch_size

                # ====== Loss Calculation ======
                with self._autocast_context():
                    logits = self.model.forward(inputs)

                if isinstance(self.loss, torch.nn.NLLLoss):
                    logits = log_softmax(logits, dim=-1)

                logits_flat = logits.contiguous().view(-1, logits.shape[-1])
                targets_flat = targets.contiguous().view(-1).clone()

                # Mask src part in targets
                seq_len = inputs.shape[1]
                ignore_index = self.loss.ignore_index
                pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)
                src_lengths_expanded = src_lengths.unsqueeze(1)
                src_mask = pos_indices < src_lengths_expanded
                targets_reshaped = targets_flat.view(batch_size, seq_len)
                targets_reshaped = targets_reshaped.masked_fill(src_mask, ignore_index)
                targets_flat = targets_reshaped.view(-1)

                loss = self.loss(logits_flat, targets_flat)
                validate_loss += loss.item() * batch_size

                # ====== BLEU Calculation (simplified like classic Transformer) ======
                if compute_bleu:
                    # Prepare left-padded source sequences
                    src_seqs, attention_mask = self._prepare_left_padded_src(inputs, src_lengths, batch_size)

                    # Generate translations
                    res = translator.translate_batch(src_seqs, attention_mask, max_new_tokens=max_new_tokens, num_beams=num_beams)

                    # Extract references
                    refer = self._extract_references(targets, src_lengths, batch_size)

                    # Calculate BLEU
                    bleu4_batch = nltk.translate.bleu_score.corpus_bleu(refer, res, weights=(0.25, 0.25, 0.25, 0.25))
                    bleu4 += bleu4_batch * batch_size

        if self.show_eval_sample and self.eval_sample_sentence:
            test_output = translator.translate(
                self.eval_sample_sentence,
                max_new_tokens=max_new_tokens,
                num_beams=self.write_config.get(
                    "eval_sample_num_beams", num_beams
                ),
            )[0]
            print(f"Sample input: {self.eval_sample_sentence}")
            print(f"Sample output: {test_output}")

        self.model.train()
        return validate_loss / sample_num, bleu4 / sample_num

    def compute_bleu_on_data(self, data_loader, unified_lang, max_new_tokens=160, num_beams=5):
        """
        Compute BLEU score on a dataset (simplified like classic Transformer)
        """
        self.model.eval()
        translator = self.get_translator(unified_lang)

        bleu4 = 0.0
        sample_num = 0

        with torch.no_grad():
            for inputs, targets, src_lengths in tqdm(data_loader, desc="Computing BLEU"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                src_lengths = src_lengths.to(self.device)
                batch_size = len(inputs)
                sample_num += batch_size

                # Prepare left-padded source sequences
                src_seqs, attention_mask = self._prepare_left_padded_src(inputs, src_lengths, batch_size)

                # Generate translations
                res = translator.translate_batch(src_seqs, attention_mask, max_new_tokens=max_new_tokens, num_beams=num_beams)

                # Extract references
                refer = self._extract_references(targets, src_lengths, batch_size)

                # Calculate BLEU
                bleu4_batch = nltk.translate.bleu_score.corpus_bleu(refer, res, weights=(0.25, 0.25, 0.25, 0.25))
                bleu4 += bleu4_batch * batch_size

        self.model.train()
        return bleu4 / sample_num

GPTTrainer = DecoderOnlyTrainer


class EncoderDecoderTrainer(BaseTrainer):
    architecture = "encoder_decoder"

    def __init__(self, model, ckpt_dir, ckpt_file_name, log_dir,
                 log_file_name, device, write_config):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.ckpt_file_name = ckpt_file_name
        self.device = device
        self.write_config = write_config
        self.optimizer = None
        self.loss = None
        self.scheduler = None
        self.global_step = 0
        self.best_loss = float("inf")
        self._configure_training_policy()
        self.log = self.make_log(log_dir, log_file_name)
        self.log.info(self.write_config)

    def build_optimizer(self, learning_rate, optimizer_type):
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate
            )
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
        else:
            raise ValueError(
                "encoder-decoder optimizer_type must be 'sgd' or 'adam'"
            )

    def build_scheduler(self, anneal_rate, scheduler_type, patience, threshold):
        if scheduler_type == "exp":
            self.scheduler = lr_scheduler.ExponentialLR(
                self.optimizer, anneal_rate
            )
        elif scheduler_type == "plateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=patience,
                factor=anneal_rate,
                threshold=threshold,
                threshold_mode="abs",
            )
        elif scheduler_type == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.write_config["max_epochs"],
                eta_min=self.write_config["learning_rate"] * 1e-2,
            )
        else:
            raise ValueError(f"unsupported scheduler_type: {scheduler_type}")

    def build_loss(self, loss_type, smoothing, ignore_index):
        if loss_type == "ce":
            self.loss = torch.nn.CrossEntropyLoss(
                ignore_index=int(ignore_index)
            )
        elif loss_type == "nll":
            self.loss = torch.nn.NLLLoss(ignore_index=int(ignore_index))
        elif loss_type == "kl":
            self.loss = LabelSmoothing(
                self.model.dec_voc_size, int(ignore_index), smoothing
            )
        else:
            raise ValueError(f"unsupported loss_type: {loss_type}")

    def fit_iter(self, source, target, accumulation_divisor=1,
                 should_update=True, clip=None):
        source = source.to(self.device)
        target = target.to(self.device)
        with self._autocast_context():
            logits = self.model(source, target[:, :-1])
        if isinstance(self.loss, (torch.nn.NLLLoss, LabelSmoothing)):
            logits = log_softmax(logits, dim=-1)
        loss = self.loss(
            logits.reshape(-1, logits.shape[-1]),
            target[:, 1:].reshape(-1),
        )
        (loss / accumulation_divisor).backward()
        if should_update:
            self._optimizer_update(clip)
        return loss.item()

    def fit(self, train_data, val_data, input_tokenizer, output_tokenizer,
            max_epochs, warmup, clip=None, **kwargs):
        for epoch in tqdm(range(self.start_epoch, max_epochs + 1)):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            if getattr(train_data, "generator", None) is not None:
                train_data.generator.manual_seed(
                    int(self.write_config.get("random_seed", 0)) + epoch
                )
            num_micro_batches = len(train_data)
            for micro_batch, (source, target) in enumerate(
                train_data, start=1
            ):
                if (
                    epoch == self.start_epoch
                    and micro_batch <= self.resume_micro_batch
                ):
                    continue
                self.micro_batch_in_epoch = micro_batch
                self.epoch_complete = False
                group_start = (
                    (micro_batch - 1) // self.gradient_accumulation_steps
                ) * self.gradient_accumulation_steps + 1
                accumulation_divisor = min(
                    self.gradient_accumulation_steps,
                    num_micro_batches - group_start + 1,
                )
                should_update = (
                    micro_batch % self.gradient_accumulation_steps == 0
                    or micro_batch == num_micro_batches
                )
                loss = self.fit_iter(
                    source,
                    target,
                    accumulation_divisor=accumulation_divisor,
                    should_update=should_update,
                    clip=clip,
                )
                losses.append(loss)
                if not should_update:
                    continue
                if self.global_step % self.print_every_n_steps == 0:
                    print(
                        f"Epoch {epoch:04d} | step {self.global_step:08d} | "
                        f"average training loss "
                        f"{sum(losses) / len(losses):.6f}"
                    )
                wandb.log(
                    {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "train_loss": loss,
                    }
                )
                if self._event_due(
                    self.save_strategy,
                    self.save_interval,
                    epoch,
                    self.global_step,
                    "step",
                ):
                    self._save_periodic_checkpoint(epoch)
                if self._event_due(
                    self.eval_strategy,
                    self.eval_interval,
                    epoch,
                    self.global_step,
                    "step",
                ):
                    self._evaluate_and_record(
                        val_data,
                        input_tokenizer,
                        output_tokenizer,
                        epoch,
                    )
                    self.model.train()

            if self._event_due(
                self.save_strategy,
                self.save_interval,
                epoch,
                self.global_step,
                "epoch",
            ):
                self.epoch_complete = True
                self._save_periodic_checkpoint(epoch)
            if self._event_due(
                self.eval_strategy,
                self.eval_interval,
                epoch,
                self.global_step,
                "epoch",
            ):
                self.epoch_complete = True
                self._evaluate_and_record(
                    val_data,
                    input_tokenizer,
                    output_tokenizer,
                    epoch,
                )
            self._step_scheduler_after_epoch(epoch, warmup)
            self.resume_micro_batch = 0
            self.epoch_complete = True

    def _evaluate_and_record(self, val_data, input_tokenizer,
                             output_tokenizer, epoch):
        val_loss, val_bleu = self.eval(
            val_data, input_tokenizer, output_tokenizer
        )
        print(
            f"Epoch {epoch:04d} | step {self.global_step:08d} | "
            f"val loss {val_loss:.6f} | BLEU-4 {val_bleu:.6f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "val_loss": val_loss,
                "val_bleu4": val_bleu,
            }
        )
        if self.save_best and val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_best_checkpoint(
                epoch,
                {"val_loss": val_loss, "val_bleu4": val_bleu},
            )
        self._step_scheduler_after_eval(val_loss)
        return val_loss, val_bleu

    @torch.no_grad()
    def eval(self, val_data, input_tokenizer, output_tokenizer):
        self.model.eval()
        requested_eval_tokens = int(
            self.write_config.get("inference_max_new_tokens", 160)
        )
        inference_max_new_tokens = min(
            requested_eval_tokens,
            int(
                getattr(
                    self.model,
                    "max_trg_len",
                    getattr(self.model, "max_context_len", 160),
                )
            ),
        )
        translator = EncoderDecoderTranslator(
            self.model,
            beam_size=self.write_config.get("beam_size", 5),
            max_seq_len=getattr(
                self.model, "max_trg_len", requested_eval_tokens + 1
            ),
            device=self.device,
            use_kv_cache=self.write_config.get(
                "inference_use_kv_cache", True
            ),
            decoding_strategy=self.write_config.get(
                "inference_decoding_strategy", "beam_search"
            ),
            temperature=self.write_config.get(
                "inference_temperature", 0.8
            ),
            top_p=self.write_config.get("inference_top_p", 0.9),
            top_k=self.write_config.get("inference_top_k", 0),
            repetition_penalty=self.write_config.get(
                "inference_repetition_penalty", 1.0
            ),
            inference_max_new_tokens=inference_max_new_tokens,
        )
        total_loss = 0.0
        sample_count = 0
        references = []
        hypotheses = []
        for source, target in tqdm(val_data, desc="Evaluating"):
            source = source.to(self.device)
            target = target.to(self.device)
            with self._autocast_context():
                logits = self.model(source, target[:, :-1])
            if isinstance(self.loss, (torch.nn.NLLLoss, LabelSmoothing)):
                logits = log_softmax(logits, dim=-1)
            loss = self.loss(
                logits.reshape(-1, logits.shape[-1]),
                target[:, 1:].reshape(-1),
            )
            batch_size = source.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            predictions = translator.translate(source)
            for expected, predicted in zip(target.tolist(), predictions):
                expected = [
                    token
                    for token in expected
                    if token not in (
                        self.model.trg_pad_idx,
                        self.model.trg_bos_idx,
                        self.model.trg_eos_idx,
                    )
                ]
                predicted = [
                    token
                    for token in predicted
                    if token not in (
                        self.model.trg_pad_idx,
                        self.model.trg_bos_idx,
                        self.model.trg_eos_idx,
                    )
                ]
                references.append([expected])
                hypotheses.append(predicted)
        bleu = corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
        )
        if self.show_eval_sample and self.eval_sample_sentence:
            source_ids = input_tokenizer.encode(
                self.eval_sample_sentence,
                character_level=input_tokenizer.name == "zh",
                add_bos=True,
                add_eos=True,
            )
            sample_source = torch.tensor(
                [source_ids], dtype=torch.long, device=self.device
            )
            sample_prediction = translator.translate(sample_source)[0]
            print(f"Sample input: {self.eval_sample_sentence}")
            print(
                "Sample output: "
                f"{output_tokenizer.decode(sample_prediction)}"
            )
        self.model.train()
        return total_loss / sample_count, bleu


def build_encoder_decoder_trainer(model, configs):
    trainer = EncoderDecoderTrainer(
        model=model,
        ckpt_dir=configs["ckpt_dir"],
        ckpt_file_name=configs["ckpt_file_name"],
        log_dir=configs["log_dir"],
        log_file_name=configs["log_file_name"],
        device=eval(configs["device"]),
        write_config=configs,
    )
    trainer.build_loss(
        configs["loss_type"],
        configs["smoothing"],
        configs["ignore_index"],
    )
    trainer.build_optimizer(
        configs["learning_rate"], configs["optimizer_type"]
    )
    if configs["scheduler_flag"]:
        trainer.build_scheduler(
            configs["anneal_rate"],
            configs["scheduler_type"],
            configs["patience"],
            configs["threshold"],
        )
    return trainer


build_trainer = build_encoder_decoder_trainer
