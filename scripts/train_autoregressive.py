"""Distributed training script for M3DLayout autoregressive model."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
from scene_synthesis.networks.autoregressive import (  # noqa: E402
    build_network,
    optimizer_factory,
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
        n_dependency_classes=train_dataset.max_length + 1,
    )
    network.to(device)
    network = torch.nn.parallel.DistributedDataParallel(
        network, device_ids=[rank], find_unused_parameters=True
    )
    network._set_static_graph()

    optimizer = optimizer_factory(
        config["training"], filter(lambda p: p.requires_grad, network.parameters())
    )

    experiment_directory = os.path.join(args.output_directory, args.experiment_tag)
    if rank == 0:
        os.makedirs(experiment_directory, exist_ok=True)
    dist.barrier()

    load_checkpoints(network, optimizer, experiment_directory, args, device)

    if rank == 0:
        save_experiment_params(args, args.experiment_tag, experiment_directory)
        if args.with_wandb_logger:
            WandB.instance().init(
                config,
                model=network.module,
                project=config.get("logger", {}).get(
                    "project", "m3dlayout_autoregressive_transformer"
                ),
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
    parser.add_argument("--config_file", default="config/m3dlayout_autoregressive.yaml")
    parser.add_argument("--output_directory", default="experiments")
    parser.add_argument("--weight_file", default=None)
    parser.add_argument("--continue_from_epoch", type=int, default=0)
    parser.add_argument("--n_processes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--experiment_tag", default="m3dlayout_autoregressive")
    parser.add_argument("--with_wandb_logger", action="store_true")
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
    parser.add_argument("--master_port", type=int, default=29501)
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

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main(sys.argv[1:])