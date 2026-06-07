import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


MODEL_FILE = "model.safetensors"
METADATA_FILE = "metadata.json"
TRAINER_STATE_FILE = "trainer_state.pt"
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _cast_state_dict(model, dtype):
    state_dict = {}
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            state_dict[name] = tensor
            continue
        if tensor.is_floating_point():
            state_dict[name] = tensor.detach().to(device="cpu", dtype=dtype)
        else:
            state_dict[name] = tensor.detach().to(device="cpu")
    return state_dict


def save_checkpoint(
    checkpoint_dir,
    model,
    metadata,
    trainer_state,
    optimizer_state=None,
    scheduler_state=None,
    is_best=False,
):
    save_precision = metadata.get("checkpoint_precision", "bf16")
    weight_dtype = (
        torch.bfloat16 if save_precision == "bf16" else torch.float32
    )
    metadata = dict(metadata)
    metadata["model_weights_dtype"] = (
        "bfloat16" if save_precision == "bf16" else "float32"
    )
    metadata["is_best"] = bool(is_best)
    metadata["model_file"] = MODEL_FILE
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = checkpoint_dir.with_name(
        f".{checkpoint_dir.name}.tmp"
    )
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()

    try:
        save_file(
            _cast_state_dict(model, weight_dtype),
            str(temporary_dir / MODEL_FILE),
        )
        with (temporary_dir / METADATA_FILE).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        torch.save(trainer_state, temporary_dir / TRAINER_STATE_FILE)
        if optimizer_state is not None:
            torch.save(optimizer_state, temporary_dir / OPTIMIZER_FILE)
        if scheduler_state is not None:
            torch.save(scheduler_state, temporary_dir / SCHEDULER_FILE)

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        os.replace(str(temporary_dir), str(checkpoint_dir))
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    return str(checkpoint_dir)


def load_checkpoint_metadata(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / METADATA_FILE).open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def load_checkpoint_model(model, checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found in {checkpoint_dir}"
        )
    state_dict = load_file(
        str(model_path),
        device=str(device),
    )
    model.load_state_dict(state_dict, strict=True)


def load_checkpoint_training_state(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    state = torch.load(
        checkpoint_dir / TRAINER_STATE_FILE,
        map_location=device,
        weights_only=False,
    )
    optimizer_path = checkpoint_dir / OPTIMIZER_FILE
    scheduler_path = checkpoint_dir / SCHEDULER_FILE
    state["optimizer_state_dict"] = (
        torch.load(
            optimizer_path, map_location=device, weights_only=False
        )
        if optimizer_path.exists()
        else None
    )
    state["scheduler_state_dict"] = (
        torch.load(
            scheduler_path, map_location=device, weights_only=False
        )
        if scheduler_path.exists()
        else None
    )
    return state
