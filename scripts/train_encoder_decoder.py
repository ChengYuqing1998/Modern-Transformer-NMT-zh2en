import argparse
import os
import pickle
import random

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, random_split

from models.transformer import build_model
from datasets.encoder_decoder import build_dataloader
from trainer.tracking import init_wandb, watch_model
from trainer.trainer import build_encoder_decoder_trainer


def parse_args():
    parser = argparse.ArgumentParser(
        "Train the encoder-decoder Transformer"
    )
    parser.add_argument(
        "--config-file-path",
        default="./configs/encoder_decoder/c2e_transformer.yaml",
    )
    return parser.parse_args()


def save_tokenizer(tokenizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(tokenizer, file)


def main():
    args = parse_args()
    with open(args.config_file_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["random_seed"])
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # Uncomment for stricter reproducibility. This can reduce performance.
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True, warn_only=True)

    def seed_worker(worker_id):
        del worker_id
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator().manual_seed(config["random_seed"])
    loader, max_seq_len, source_tokenizer, target_tokenizer, _ = build_dataloader(
        "zh",
        "en",
        config["max_target_sentence_split_length"],
        config["min_sequence_token_length"],
        config["batch_size"],
        seed_worker,
        generator,
    )

    tokenizer_paths = config["tokenizer_paths"]
    save_tokenizer(source_tokenizer, tokenizer_paths["source"])
    save_tokenizer(target_tokenizer, tokenizer_paths["target"])

    train_length = int(config["train_ratio"] * len(loader.dataset))
    val_length = int(config["val_ratio"] * len(loader.dataset))
    test_length = len(loader.dataset) - train_length - val_length
    train_set, val_set, test_set = random_split(
        loader.dataset,
        [train_length, val_length, test_length],
        generator=generator,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False
    )

    init_wandb(config, default_project="modern-transformer-zh-en")
    max_src_len = max_seq_len
    max_trg_len = max_seq_len
    model = build_model(
        source_tokenizer, target_tokenizer, max_src_len, max_trg_len, config
    ).to(eval(config["device"], {"torch": torch}))
    watch_model(model, config)
    trainer = build_encoder_decoder_trainer(model, config)
    if config.get("resume_from_checkpoint"):
        trainer.resume_from_checkpoint(config["resume_from_checkpoint"])
    clip = config["clip_norm"] if config.get("clip_flag") else None
    trainer.fit(
        train_loader,
        val_loader,
        source_tokenizer,
        target_tokenizer,
        config["max_epochs"],
        warmup=config["warmup"],
        clip=clip,
    )
    test_loss, test_bleu = trainer.eval(
        test_loader, source_tokenizer, target_tokenizer
    )
    print(f"Test loss: {test_loss:.6f}; BLEU-4: {test_bleu:.6f}")
    wandb.log({"test_loss": test_loss, "test_bleu4": test_bleu})
    wandb.finish()


if __name__ == "__main__":
    main()
