import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import json

import string
import os
import random
import torch


def _resolve_path(path, base_dir):
    if path is None:
        return None
    if isinstance(path, str) and not os.path.isabs(path):
        return os.path.normpath(os.path.join(base_dir, path))
    return path


def _normalize_config_paths(config, config_path):
    """Normalize relative paths in config to absolute paths.

    This makes the training scripts robust to different working directories.
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for split_key in ("train_data", "val_data"):
        split_cfg = config.get(split_key, {})
        if "train_stats_file" in split_cfg:
            split_cfg["train_stats_file"] = _resolve_path(
                split_cfg["train_stats_file"], config_dir
            )
        if "json_files" in split_cfg:
            split_cfg["json_files"] = [
                _resolve_path(path, config_dir) for path in split_cfg["json_files"]
            ]

    diffusion_kwargs = config.get("network", {}).get("diffusion_kwargs", {})
    train_stats_file = diffusion_kwargs.get("train_stats_file")
    if isinstance(train_stats_file, str):
        diffusion_kwargs["train_stats_file"] = _resolve_path(train_stats_file, config_dir)


def load_config(config_file):
    config_path = os.path.abspath(config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    _normalize_config_paths(config, config_path)
    return config


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


#### transferred from train_network.py
def yield_forever(iterator):
    while True:
        for x in iterator:
            yield x


def _get_checkpoint_model(model):
    return model.module if hasattr(model, "module") else model


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    if not os.path.isdir(experiment_directory):
        return

    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_") and f[6:].isdigit()
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}"
    ).format(max_id)
    opt_path = os.path.join(
        experiment_directory, "opt_{:05d}"
    ).format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    state_dict = torch.load(model_path, map_location=device)
    model_to_load = _get_checkpoint_model(model)
    missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
    if missing:
        print("WARNING: missing model keys ({}).".format(len(missing)))
    if unexpected:
        print("WARNING: unexpected model keys ({}).".format(len(unexpected)))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    model_to_save = _get_checkpoint_model(model)
    torch.save(
        model_to_save.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )