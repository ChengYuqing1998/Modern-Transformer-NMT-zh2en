import argparse
import pickle
from pathlib import Path

from datasets.corpus import prepare_parallel_corpus
from tokenizer.tokenizer import Vocabulary


def save_tokenizer(tokenizer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(tokenizer, file)


def build_encoder_decoder(data_path, output_dir, max_target_length):
    pairs, source_language, target_language = prepare_parallel_corpus(
        "zh",
        "en",
        data_path=data_path,
        max_target_length=max_target_length,
    )
    source = Vocabulary(source_language)
    target = Vocabulary(target_language)
    for source_text, target_text in pairs:
        source.add_text(
            source_text, character_level=source_language == "zh"
        )
        target.add_text(
            target_text, character_level=target_language == "zh"
        )
    save_tokenizer(source, Path(output_dir) / "source_tokenizer.pkl")
    save_tokenizer(target, Path(output_dir) / "target_tokenizer.pkl")
    return {"source": len(source), "target": len(target)}


def build_decoder_only(data_path, output_dir, max_target_length):
    pairs, source_language, target_language = prepare_parallel_corpus(
        "zh",
        "en",
        data_path=data_path,
        max_target_length=max_target_length,
    )
    unified = Vocabulary("mixed")
    for source_text, target_text in pairs:
        unified.add_text(
            source_text, character_level=source_language == "zh"
        )
        unified.add_text(
            target_text, character_level=target_language == "zh"
        )
    save_tokenizer(unified, Path(output_dir) / "unified_tokenizer.pkl")
    return {"unified": len(unified)}


def parse_args():
    parser = argparse.ArgumentParser("Build translation tokenizers")
    parser.add_argument(
        "--architecture",
        choices=("encoder_decoder", "decoder_only", "all"),
        default="all",
    )
    parser.add_argument("--data-path", default="./data/zh-en.txt")
    parser.add_argument("--output-dir", default="./tokenizer/artifacts")
    parser.add_argument("--max-target-length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    sizes = {}
    if args.architecture in ("encoder_decoder", "all"):
        sizes.update(
            build_encoder_decoder(
                args.data_path, args.output_dir, args.max_target_length
            )
        )
    if args.architecture in ("decoder_only", "all"):
        sizes.update(
            build_decoder_only(
                args.data_path, args.output_dir, args.max_target_length
            )
        )
    print("Built tokenizers:", ", ".join(f"{name}={size}" for name, size in sizes.items()))


if __name__ == "__main__":
    main()
