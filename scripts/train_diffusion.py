"""Distributed training script for M3DLayout diffusion model."""
import argparse
import os
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from training_utils import (  # noqa: E402
    load_checkpoints,
    load_config,
    save_checkpoints,
    save_experiment_params,
)
from scene_synthesis.datasets import M3DLayoutDataset  # noqa: E402
from scene_synthesis.networks.diffusion import (  # noqa: E402
    adjust_learning_rate,
    build_network,
    optimizer_factory,
    schedule_factory,
)
from scene_synthesis.stats_logger import StatsLogger, WandB  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _parse_source_weights(source_weights):
    weights = {}
    for item in source_weights.split(","):
        source, weight = item.strip().split(":")
        weights[source.strip().lower()] = float(weight)
    return weights


def _attach_source_weights(sample, source_weights_dict, device):
    if source_weights_dict is None:
        return

    batch_source_weights = []
    if "data_source" in sample:
        for source in sample["data_source"]:
            source_name = source.lower() if isinstance(source, str) else str(source).lower()
            weight = 1.0
            for key, val in source_weights_dict.items():
                if key in source_name:
                    weight = val
                    break
            batch_source_weights.append(weight)
    else:
        batch_size = sample["class_labels"].shape[0]
        batch_source_weights = [1.0] * batch_size

    sample["source_weights"] = torch.tensor(batch_source_weights, device=device)


def update_config_paths(config, backup_dir):
    """Update paths in config to point to copied files under backup dir."""
    updated_config = deepcopy(config)
    datasets_dir = os.path.join(backup_dir, "datasets")

    for split_key in ("train_data", "val_data"):
        split_cfg = updated_config.get(split_key, {})
        if "train_stats_file" in split_cfg:
            filename = os.path.basename(split_cfg["train_stats_file"])
            split_cfg["train_stats_file"] = os.path.join(datasets_dir, filename)
        if "json_files" in split_cfg:
            split_cfg["json_files"] = [
                os.path.join(datasets_dir, os.path.basename(p)) for p in split_cfg["json_files"]
            ]

    diffusion_kwargs = updated_config.get("network", {}).get("diffusion_kwargs", {})
    if isinstance(diffusion_kwargs.get("train_stats_file"), str):
        filename = os.path.basename(diffusion_kwargs["train_stats_file"])
        diffusion_kwargs["train_stats_file"] = os.path.join(datasets_dir, filename)

    return updated_config


def copy_experiment_files(experiment_directory, args, config):
    """Copy code/config/dataset files into experiment directory for reproducibility."""
    print(f"Copying experiment files to {experiment_directory}")

    backup_dir = os.path.join(experiment_directory, "experiment_backup")
    os.makedirs(backup_dir, exist_ok=True)

    scene_synthesis_src = os.path.join(PROJECT_ROOT, "scene_synthesis")
    scene_synthesis_dst = os.path.join(backup_dir, "scene_synthesis")
    if os.path.exists(scene_synthesis_src):
        shutil.copytree(scene_synthesis_src, scene_synthesis_dst, dirs_exist_ok=True)
        print(f"Copied scene_synthesis to {scene_synthesis_dst}")

    scripts_dir = os.path.join(backup_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    current_script = os.path.abspath(__file__)
    shutil.copy2(current_script, os.path.join(scripts_dir, os.path.basename(current_script)))

    training_utils_src = os.path.join(os.path.dirname(current_script), "training_utils.py")
    if os.path.exists(training_utils_src):
        shutil.copy2(training_utils_src, os.path.join(scripts_dir, "training_utils.py"))

    config_src = args.config_file
    if os.path.exists(config_src):
        config_dst = os.path.join(backup_dir, "config.yaml")
        updated_config = update_config_paths(config, backup_dir)
        with open(config_dst, "w") as f:
            yaml.dump(updated_config, f, default_flow_style=False, indent=2)

        original_config_dst = os.path.join(backup_dir, "config_original.yaml")
        shutil.copy2(config_src, original_config_dst)

    datasets_dst = os.path.join(backup_dir, "datasets")
    os.makedirs(datasets_dst, exist_ok=True)

    dataset_paths = set()
    for split_key in ("train_data", "val_data"):
        split_cfg = config.get(split_key, {})
        if "train_stats_file" in split_cfg:
            dataset_paths.add(split_cfg["train_stats_file"])
        for json_file in split_cfg.get("json_files", []):
            dataset_paths.add(json_file)

    diffusion_stats = config.get("network", {}).get("diffusion_kwargs", {}).get("train_stats_file")
    if isinstance(diffusion_stats, str):
        dataset_paths.add(diffusion_stats)

    copied_count = 0
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            dst = os.path.join(datasets_dst, os.path.basename(dataset_path))
            shutil.copy2(dataset_path, dst)
            copied_count += 1
        else:
            print(f"Warning: Dataset file not found: {dataset_path}")

    print(f"Copied {copied_count} dataset files to {datasets_dst}")
    print("Experiment files backup completed!")


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    config = load_config(args.config_file)

    train_dataset = M3DLayoutDataset(config["dataset"] | config["train_data"])
    val_dataset = M3DLayoutDataset(config["dataset"] | config["val_data"])

    source_weights_dict = None
    if args.balance_loss:
        source_weights_dict = _parse_source_weights(args.source_weights)
        if rank == 0:
            print(f"Data source weights for loss balancing: {source_weights_dict}")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    persistent_workers = args.n_processes > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 32),
        sampler=train_sampler,
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        sampler=val_sampler,
        num_workers=args.n_processes,
        collate_fn=val_dataset.collate_fn,
        persistent_workers=persistent_workers,
    )

    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.feature_size,
        train_dataset.n_classes,
        config,
        args.weight_file,
        device=str(device),
    )
    network.to(device)
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[rank])

    optimizer = optimizer_factory(
        config["training"], filter(lambda p: p.requires_grad, network.parameters())
    )
    lr_scheduler = schedule_factory(config["training"])

    experiment_directory = os.path.join(args.output_directory, args.experiment_tag)
    if rank == 0:
        os.makedirs(experiment_directory, exist_ok=True)
    dist.barrier()

    load_checkpoints(network, optimizer, experiment_directory, args, device)

    if rank == 0:
        save_experiment_params(args, args.experiment_tag, experiment_directory)

        if args.backup_files:
            copy_experiment_files(experiment_directory, args, config)

        if args.with_wandb_logger:
            WandB.instance().init(
                config,
                model=network.module,
                project=config.get("logger", {}).get("project", "m3dlayout_diffusion"),
                name=args.experiment_tag,
                watch=False,
                log_frequency=10,
            )
        StatsLogger.instance().add_output_file(
            open(os.path.join(experiment_directory, "stats.txt"), "w")
        )

    epochs = config["training"].get("epochs", 150)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 10)

    if rank == 0:
        StatsLogger.instance().set_total_epochs(epochs)

    for epoch in range(args.continue_from_epoch, epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(lr_scheduler, optimizer, epoch)
        network.train()

        for b, sample in enumerate(train_loader):
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(device)

            _attach_source_weights(sample, source_weights_dict, device)

            batch_loss = train_on_batch(network, optimizer, sample, config)
            if rank == 0:
                StatsLogger.instance().print_progress(epoch + 1, b + 1, batch_loss)

        if rank == 0 and (epoch % save_every == 0):
            save_checkpoints(epoch, network, optimizer, experiment_directory)

        if rank == 0:
            StatsLogger.instance().clear()

        if epoch % val_every == 0 and epoch > 0:
            network.eval()
            for b, sample in enumerate(val_loader):
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        sample[k] = v.to(device)

                _attach_source_weights(sample, source_weights_dict, device)

                batch_loss = validate_on_batch(network, sample, config)
                if rank == 0:
                    StatsLogger.instance().print_progress(-1, b + 1, batch_loss)
            if rank == 0:
                StatsLogger.instance().clear()

    if rank == 0:
        print("Training complete!")

    dist.destroy_process_group()


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/m3dlayout_diffusion.yaml")
    parser.add_argument("--output_directory", default="experiments")
    parser.add_argument("--weight_file", default=None)
    parser.add_argument("--continue_from_epoch", type=int, default=0)
    parser.add_argument("--n_processes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--experiment_tag", default="m3dlayout_diffusion")
    parser.add_argument("--with_wandb_logger", action="store_true")
    parser.add_argument(
        "--backup_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backup experiment files (code, config, dataset info) to experiment directory",
    )
    parser.add_argument(
        "--balance_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable balancing loss weights by data source",
    )
    parser.add_argument(
        "--source_weights",
        type=str,
        default="3dfront:2.0,mp3d:1.0,infinigen:1.0",
        help="Weight multipliers for different data sources, format: 'source1:weight1,source2:weight2'",
    )
    parser.add_argument("--master_port", type=int, default=29502)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    args.config_file = _resolve_path(args.config_file)
    args.output_directory = _resolve_path(args.output_directory)
    if args.weight_file:
        args.weight_file = _resolve_path(args.weight_file)

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found. DDP training requires at least one GPU.")

    print(f"Using {world_size} GPUs for training")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port or random.randint(20000, 30000)))
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main(sys.argv[1:])