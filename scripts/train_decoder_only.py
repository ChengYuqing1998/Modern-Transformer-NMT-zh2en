import argparse
import os
import pickle
import random

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, random_split

from models.transformer import build_GPT
from datasets.decoder_only import build_dataloader
from trainer.tracking import init_wandb, watch_model
from trainer.trainer import build_decoder_only_trainer


def parse_args():
    parser = argparse.ArgumentParser("Train the decoder-only Transformer")
    parser.add_argument(
        "--config-file-path",
        default="./configs/decoder_only/c2e_gpt.yaml",
    )
    return parser.parse_args()


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
    loader, sequence_token_length, unified_tokenizer, _ = build_dataloader(
        "zh",
        "en",
        config["max_target_sentence_split_length"],
        config["max_context_len"],
        config["min_sequence_token_length"],
        config["batch_size"],
        seed_worker,
        generator,
    )
    tokenizer_path = config["tokenizer_paths"]["unified"]
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, "wb") as file:
        pickle.dump(unified_tokenizer, file)
    del sequence_token_length

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
    model = build_GPT(
        unified_tokenizer, config["max_context_len"], config
    ).to(eval(config["device"], {"torch": torch}))
    watch_model(model, config)
    trainer = build_decoder_only_trainer(model, config)
    if config.get("resume_from_checkpoint"):
        trainer.resume_from_checkpoint(config["resume_from_checkpoint"])
    clip = config["clip_norm"] if config.get("clip_flag") else None
    trainer.fit(
        train_loader,
        val_loader,
        unified_tokenizer,
        config["max_epochs"],
        warmup=config["warmup"],
        clip=clip,
    )
    test_bleu = trainer.compute_bleu_on_data(
        test_loader,
        unified_tokenizer,
        max_new_tokens=config.get("inference_max_new_tokens", 160),
        num_beams=config.get("beam_size", 5),
    )
    print(f"Test BLEU-4: {test_bleu:.6f}")
    wandb.log({"test_bleu4": test_bleu})
    wandb.finish()


if __name__ == "__main__":
    main()
