#!/usr/bin/env python
"""Build a submission-ready Soccer-Twos agent module from a checkpoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = sorted(child for child in path.iterdir() if child.name.startswith("checkpoint-"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* file found in directory: {path}")
    return candidates[0]


def find_params_pkl(checkpoint_file: Path) -> Path:
    config_dir = checkpoint_file.parent
    candidates = [
        config_dir / "params.pkl",
        config_dir.parent / "params.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find params.pkl near checkpoint: {checkpoint_file}")


def infer_module_import_path(output_dir: Path) -> str:
    cwd = Path.cwd().resolve()
    try:
        relative = output_dir.resolve().relative_to(cwd)
    except ValueError:
        return output_dir.name
    return ".".join(relative.parts)


def render_readme(
    readme_path: Path,
    *,
    module_name: str,
    import_path: str,
    agent_name: str,
    author: str,
    description: str,
    snapshot: str,
) -> None:
    text = readme_path.read_text(encoding="utf-8")
    text = text.replace("_版本名_", module_name)
    text = text.replace("_填写_", agent_name, 1)
    text = text.replace("_填写姓名_ (_填写邮箱_)", author)
    text = text.replace("_训练方法、reward 设计、关键超参等_", description)
    text = text.replace("snapshot-NNN", snapshot)
    text = text.replace("agents.vNNN_xxx", import_path)
    readme_path.write_text(text, encoding="utf-8")


def build_agent_module(
    *,
    checkpoint: Path,
    output_dir: Path,
    template_dir: Path,
    agent_name: str,
    author: str,
    description: str,
    snapshot: str,
    overwrite: bool,
    make_zip: bool,
) -> None:
    checkpoint_file = resolve_checkpoint_file(checkpoint.resolve())
    params_file = find_params_pkl(checkpoint_file)
    template_dir = template_dir.resolve()
    output_dir = output_dir.resolve()

    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                f"Use --force to overwrite."
            )
        shutil.rmtree(output_dir)

    shutil.copytree(template_dir, output_dir)
    shutil.copy2(checkpoint_file, output_dir / "checkpoint")
    shutil.copy2(params_file, output_dir / "params.pkl")
    import_path = infer_module_import_path(output_dir)

    readme_path = output_dir / "README.md"
    if readme_path.exists():
        render_readme(
            readme_path,
            module_name=output_dir.name,
            import_path=import_path,
            agent_name=agent_name,
            author=author,
            description=description,
            snapshot=snapshot,
        )

    required_files = ("__init__.py", "agent.py", "README.md", "checkpoint", "params.pkl")
    missing = [name for name in required_files if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Built module is missing required files: {missing}")

    print(f"Built agent module: {output_dir}")
    print(f"  checkpoint: {checkpoint_file}")
    print(f"  params.pkl:  {params_file}")

    if make_zip:
        archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
        print(f"  zip:         {archive_path}")

    print("")
    print("Next steps:")
    print(f"  python -m cs8803drl.evaluation.evaluate_matches -m1 {import_path} -m2 ceia_baseline_agent -n 10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint file or checkpoint directory")
    parser.add_argument("--output-dir", "-o", required=True, help="Destination agent module directory")
    parser.add_argument(
        "--template-dir",
        default="agents/_template",
        help="Template directory to copy before injecting checkpoint artifacts",
    )
    parser.add_argument("--agent-name", default="RayAgent", help="Display name written into README")
    parser.add_argument("--author", default="TBD", help="Author line written into README")
    parser.add_argument(
        "--description",
        default="PPO policy exported from RLlib checkpoint.",
        help="Short description written into README",
    )
    parser.add_argument(
        "--snapshot",
        default="snapshot-TBD",
        help="Experiment snapshot reference written into README",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it already exists")
    parser.add_argument("--zip", action="store_true", help="Also create <output-dir>.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_agent_module(
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        template_dir=Path(args.template_dir),
        agent_name=args.agent_name,
        author=args.author,
        description=args.description,
        snapshot=args.snapshot,
        overwrite=bool(args.force),
        make_zip=bool(args.zip),
    )


if __name__ == "__main__":
    main()
