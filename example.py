"""Batch inference script for M3DLayout.

Reads prompts from text files and runs one inference per line.
Each input text file produces one JSON output file that is overwritten
after every processed prompt so progress can be monitored live.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from gradio_demo import M3DLayoutArGenerator, M3DLayoutDiffusionGenerator


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_FILES = [
    "examples/desc_bedroom_500.txt",
    "examples/desc_dining_room_500.txt",
    "examples/desc_living_room_500.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate M3DLayout scenes from prompt text files."
    )
    parser.add_argument(
        "--model",
        choices=["autoregressive", "diffusion"],
        default="autoregressive",
        help="Model type used for generation.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=DEFAULT_INPUT_FILES,
        help="Prompt text files to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch_results",
        help="Directory to store JSON outputs and optional GIFs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for each prompt.",
    )
    parser.add_argument(
        "--render-gif",
        dest="render_gif",
        action="store_true",
        help="Render and save viz3dl GIF for each prompt.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_input_files(input_files: List[str]) -> List[Path]:
    resolved_files: List[Path] = []
    for input_file in input_files:
        input_path = resolve_path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        resolved_files.append(input_path)
    return resolved_files


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    return value


def write_json_overwrite(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    temp_path.replace(output_path)


def load_prompts(input_path: Path) -> List[str]:
    with input_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_generator(model: str, seed: int):
    if model == "diffusion":
        return M3DLayoutDiffusionGenerator(
            config_file=str(PROJECT_ROOT / "config/m3dlayout_diffusion.yaml"),
            weight_file=str(PROJECT_ROOT / "weights/diffusion_30000.pth"),
            default_seed=seed,
        )
    return M3DLayoutArGenerator(
        config_file=str(PROJECT_ROOT / "config/m3dlayout_autoregressive.yaml"),
        weight_file=str(PROJECT_ROOT / "weights/autoregressive_59000.pth"),
        default_seed=seed,
    )


def process_file(
    generator,
    input_path: Path,
    output_dir: Path,
    model: str,
    seed: int,
    render_gif: bool,
) -> Path:
    prompts = load_prompts(input_path)

    output_json_path = output_dir / f"{input_path.stem}_results.json"
    gif_dir = output_dir / f"{input_path.stem}_gifs"
    if render_gif:
        gif_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    success_count = 0
    failed_count = 0

    payload: Dict[str, Any] = {
        "input_file": str(input_path),
        "output_file": str(output_json_path),
        "model": model,
        "seed": seed,
        "render_gif": render_gif,
        "total_prompts": len(prompts),
        "processed_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "started_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "results": results,
    }
    write_json_overwrite(output_json_path, payload)

    for idx, prompt in enumerate(prompts, start=1):
        line_start = time.perf_counter()
        result_item: Dict[str, Any] = {
            "index": idx,
            "prompt": prompt,
            "seed": seed,
            "status": "pending",
            "processed_at": "",
            "elapsed_sec": 0.0,
        }

        try:
            if not prompt.strip():
                raise ValueError("Prompt is empty")

            scene_data = generator.generate_scene_from_text(prompt, seed=seed)
            result_item["status"] = "success"
            result_item["scene"] = to_jsonable(scene_data)

            if render_gif:
                gif_path = gif_dir / f"{input_path.stem}_{idx:04d}.gif"
                generator.visualize_scene(scene_data, str(gif_path))
                result_item["gif_path"] = str(gif_path)

            success_count += 1
        except Exception as e:
            failed_count += 1
            result_item["status"] = "failed"
            result_item["error"] = str(e)

        result_item["elapsed_sec"] = round(time.perf_counter() - line_start, 4)
        result_item["processed_at"] = datetime.now().isoformat()

        results.append(result_item)

        payload["processed_count"] = idx
        payload["success_count"] = success_count
        payload["failed_count"] = failed_count
        payload["last_updated"] = datetime.now().isoformat()
        write_json_overwrite(output_json_path, payload)

        print(
            f"[{input_path.name}] {idx}/{len(prompts)} | "
            f"status={result_item['status']} | "
            f"success={success_count} fail={failed_count}"
        )

    return output_json_path


def main() -> None:
    args = parse_args()

    input_files = resolve_input_files(args.input_files)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = build_generator(args.model, args.seed)

    output_paths = []
    for input_path in input_files:
        print(f"Processing prompts from: {input_path}")
        output_json_path = process_file(
            generator=generator,
            input_path=input_path,
            output_dir=output_dir,
            model=args.model,
            seed=args.seed,
            render_gif=args.render_gif,
        )
        output_paths.append(output_json_path)
        print(f"Saved results to: {output_json_path}")

    print("Batch inference finished.")
    for output_path in output_paths:
        print(f" - {output_path}")


if __name__ == "__main__":
    main()