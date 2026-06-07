import os

import wandb


VALID_WANDB_MODES = {"disabled", "offline", "online"}


def resolve_wandb_settings(config, default_project):
    mode = os.getenv("WANDB_MODE", config.get("wandb_mode", "disabled"))
    if mode not in VALID_WANDB_MODES:
        raise ValueError(
            "wandb_mode must be 'disabled', 'offline', or 'online'"
        )
    return {
        "mode": mode,
        "project": os.getenv(
            "WANDB_PROJECT",
            config.get("wandb_project", default_project),
        ),
        "entity": os.getenv("WANDB_ENTITY") or None,
    }


def init_wandb(config, default_project):
    settings = resolve_wandb_settings(config, default_project)
    run = wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        name=str(config["trial_name"]),
        config=config,
        mode=settings["mode"],
    )
    wandb.define_metric("global_step")
    for metric_name in (
        "train_loss",
        "average_train_loss",
        "learning_rate",
        "val_loss",
        "val_bleu4",
    ):
        wandb.define_metric(metric_name, step_metric="global_step")
    print(
        f"W&B mode: {settings['mode']}; "
        f"project: {settings['project']}"
    )
    return run


def watch_model(model, config):
    if config.get("wandb_watch_model", False):
        wandb.watch(model)
