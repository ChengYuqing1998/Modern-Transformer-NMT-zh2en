import argparse
import pickle
from pathlib import Path

import torch

from inference.translator import (
    DecoderOnlyTranslator,
    EncoderDecoderTranslator,
)
from models.transformer import build_GPT, build_model
from tokenizer.tokenizer import normalize_string
from trainer.checkpoint import (
    load_checkpoint_metadata,
    load_checkpoint_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Translate with an encoder-decoder or decoder-only checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sentence")
    parser.add_argument(
        "--architecture",
        choices=("auto", "encoder_decoder", "decoder_only"),
        default="auto",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--beam-size", type=int)
    parser.add_argument(
        "--inference-max-new-tokens",
        dest="inference_max_new_tokens",
        type=int,
        help=(
            "Override inference_max_new_tokens stored in checkpoint metadata."
        ),
    )
    parser.add_argument(
        "--decoding-strategy",
        choices=("greedy", "beam_search", "nucleus_sampling"),
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--use-kv-cache",
        dest="use_kv_cache",
        action="store_true",
        help="Enable incremental KV-cache decoding.",
    )
    cache_group.add_argument(
        "--no-kv-cache",
        dest="use_kv_cache",
        action="store_false",
        help="Recompute the full generated prefix at every decoding step.",
    )
    parser.set_defaults(use_kv_cache=None)
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_runtime(
    checkpoint_path,
    requested_architecture,
    device,
    beam_size=None,
    use_kv_cache=None,
    decoding_strategy=None,
):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint = load_checkpoint_metadata(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if (
            not isinstance(checkpoint, dict)
            or "model_state_dict" not in checkpoint
        ):
            raise ValueError(
                "Unsupported checkpoint format. Pass a checkpoint directory "
                "or a legacy state_dict checkpoint."
            )
    architecture = checkpoint.get("architecture")
    if requested_architecture != "auto":
        architecture = requested_architecture
    config = dict(checkpoint["model_config"])
    config["_device"] = str(device)
    tokenizer_paths = checkpoint.get("tokenizer_paths") or config.get(
        "tokenizer_paths", {}
    )
    max_src_len = checkpoint.get("max_src_len")
    max_trg_len = checkpoint.get("max_trg_len")
    max_context_len = checkpoint.get("max_context_len")
    resolved_use_kv_cache = (
        config.get("inference_use_kv_cache", True)
        if use_kv_cache is None
        else use_kv_cache
    )

    if architecture == "encoder_decoder":
        resolved_decoding_strategy = decoding_strategy or config.get(
            "inference_decoding_strategy", "beam_search"
        )
        resolved_beam_size = (
            1
            if resolved_decoding_strategy == "greedy"
            else beam_size or config.get("beam_size", 5)
        )
        if max_src_len is None or max_trg_len is None:
            raise ValueError(
                "Encoder-decoder checkpoints must include max_src_len and max_trg_len."
            )
        source_tokenizer = load_pickle(tokenizer_paths["source"])
        target_tokenizer = load_pickle(tokenizer_paths["target"])
        model = build_model(
            source_tokenizer,
            target_tokenizer,
            max_src_len,
            max_trg_len,
            config,
        )
        if checkpoint_path.is_dir():
            load_checkpoint_model(model, checkpoint_path, device)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        translator = EncoderDecoderTranslator(
            model,
            beam_size=resolved_beam_size,
            max_seq_len=max_trg_len,
            device=device,
            use_kv_cache=resolved_use_kv_cache,
            decoding_strategy=resolved_decoding_strategy,
            temperature=config.get("inference_temperature", 0.8),
            top_p=config.get("inference_top_p", 0.9),
            top_k=config.get("inference_top_k", 0),
            repetition_penalty=config.get(
                "inference_repetition_penalty", 1.0
            ),
            inference_max_new_tokens=config.get(
                "inference_max_new_tokens", 160
            ),
        )
        return architecture, translator, source_tokenizer, target_tokenizer

    if architecture == "decoder_only":
        if max_context_len is None:
            raise ValueError(
                "Decoder-only checkpoints must include max_context_len."
            )
        tokenizer = load_pickle(tokenizer_paths["unified"])
        model = build_GPT(tokenizer, max_context_len, config)
        if checkpoint_path.is_dir():
            load_checkpoint_model(model, checkpoint_path, device)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        translator = DecoderOnlyTranslator(
            model,
            tokenizer,
            device=device,
            use_kv_cache=resolved_use_kv_cache,
            decoding_strategy=config.get(
                "inference_decoding_strategy", "beam_search"
            ),
            num_beams=beam_size or config.get("beam_size", 5),
            temperature=config.get("inference_temperature", 0.8),
            top_p=config.get("inference_top_p", 0.9),
            top_k=config.get("inference_top_k", 0),
            repetition_penalty=config.get(
                "inference_repetition_penalty", 1.0
            ),
            inference_max_new_tokens=config.get(
                "inference_max_new_tokens", 160
            ),
        )
        return architecture, translator, tokenizer, tokenizer

    raise ValueError(
        f"Unsupported architecture {architecture!r}. "
        "The VLM does not yet provide a translation generation pipeline."
    )


def translate_sentence(
    architecture,
    translator,
    source_tokenizer,
    target_tokenizer,
    sentence,
    beam_size,
    max_new_tokens,
    decoding_strategy=None,
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=None,
):
    sentence = normalize_string(sentence)
    if architecture == "decoder_only":
        return translator.translate(
            sentence,
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            num_beams=beam_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )[0]

    token_ids = source_tokenizer.encode(
        sentence,
        character_level=source_tokenizer.name == "zh",
        add_bos=True,
        add_eos=True,
    )
    source = torch.tensor(
        [token_ids], dtype=torch.long, device=translator.device
    )
    prediction = translator.translate(
        source,
        max_new_tokens=max_new_tokens,
        decoding_strategy=decoding_strategy,
        num_beams=beam_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )[0]
    return target_tokenizer.decode(prediction)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    runtime = load_runtime(
        args.checkpoint,
        args.architecture,
        device,
        args.beam_size,
        args.use_kv_cache,
        args.decoding_strategy,
    )
    architecture, translator, source_tokenizer, target_tokenizer = runtime

    if args.sentence:
        print(
            translate_sentence(
                architecture,
                translator,
                source_tokenizer,
                target_tokenizer,
                args.sentence,
                args.beam_size,
                args.inference_max_new_tokens,
                args.decoding_strategy,
                args.temperature,
                args.top_p,
                args.top_k,
                args.repetition_penalty,
            )
        )
        return

    while True:
        sentence = input("Chinese sentence (empty input exits): ").strip()
        if not sentence:
            break
        print(
            translate_sentence(
                architecture,
                translator,
                source_tokenizer,
                target_tokenizer,
                sentence,
                args.beam_size,
                args.inference_max_new_tokens,
                args.decoding_strategy,
                args.temperature,
                args.top_p,
                args.top_k,
                args.repetition_penalty,
            )
        )


if __name__ == "__main__":
    main()
