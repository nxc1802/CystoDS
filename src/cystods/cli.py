"""CystoDS unified command-line interface.

Usage:
    cystods run <stage> [--profile P] [--config C] [--set key=val ...]
    cystods config show [--stage S] [--profile P] [--config C]
    cystods stages
    cystods validate [--config C] [--profile P]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_STAGE_REGISTRY: dict[str, str] = {
    "00": "Prepare protocol — data audit + freeze patient-disjoint split",
    "10": "Run baselines — binary/coarse/fine/multitask across 4 backbones",
    "20": "Long-tail loss screen — 7 fine-only loss variants on Swin-Tiny",
    "30": "Proposed Method (3S-HFT) — Sequential 3-Stage Hierarchical Fine-Tuning",
    "40": "Ablation studies — 8 Component, Paradigm, Strategy & Freezing Variants",
    "60": "External validation — evaluation-only on external cohort",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cystods",
        description="CystoDS: Hierarchical Long-Tailed Cystoscopy Classification",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- cystods run ---
    run_parser = subparsers.add_parser("run", help="Run a pipeline stage")
    run_parser.add_argument(
        "stage",
        type=str,
        help="Stage number (00, 10, 20, 30, 40, 60, 90) or 'all'",
    )
    run_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Run profile: research (default) or smoke",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values: --set key=value",
    )
    run_parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Protocol split index ('0', '1', '2', or 'all'). Default: all",
    )
    run_parser.add_argument(
        "--models",
        "--model",
        nargs="+",
        dest="models",
        default=None,
        help="Filter trials by model backbone(s) or alias (e.g. swin_tiny, resnet152, hrnet_w18, resnext50_32x4d)",
    )
    run_parser.add_argument(
        "--trials",
        "--trial",
        "--experiments",
        "--experiment",
        nargs="+",
        dest="trials",
        default=None,
        help="Filter trials by experiment ID(s) or pattern (e.g. binary_swin_tiny, coarse_swin_tiny)",
    )
    run_parser.add_argument(
        "--partial-finetune",
        "--freeze-early-layers",
        "--freeze-stage2",
        dest="partial_finetune",
        action="store_true",
        default=False,
        help="Freeze Patch Embedding + Stage 1 + Stage 2; only fine-tune Stage 3, Stage 4 and classification heads",
    )
    run_parser.add_argument(
        "--freeze-stage3",
        "--freeze-through-stage3",
        dest="freeze_stage3",
        action="store_true",
        default=False,
        help="Freeze Patch Embedding + Stage 1 + Stage 2 + Stage 3; only fine-tune Stage 4 and classification heads (~50%% compute reduction)",
    )
    run_parser.add_argument(
        "--freeze-stages",
        "--frozen-stages",
        type=int,
        choices=[1, 2, 3],
        default=None,
        dest="freeze_stages",
        help="Explicitly specify number of early backbone stages to freeze (1: Stage 1, 2: Stages 1-2, 3: Stages 1-3)",
    )

    # --- cystods config ---
    config_parser = subparsers.add_parser("config", help="Show resolved config")
    config_sub = config_parser.add_subparsers(dest="config_command")
    show_parser = config_sub.add_parser("show", help="Display resolved config")
    show_parser.add_argument("--stage", type=str, default=None)
    show_parser.add_argument("--split", type=int, choices=[0, 1, 2], default=None)
    show_parser.add_argument("--profile", type=str, default=None)
    show_parser.add_argument("--config", type=str, default=None)
    show_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values: --set key=value",
    )

    # --- cystods stages ---
    subparsers.add_parser("stages", help="List available pipeline stages")

    # --- cystods run-remaining ---
    remaining_parser = subparsers.add_parser(
        "run-remaining",
        help="Run all remaining pipeline stages (30 -> 40) sequentially",
    )
    remaining_parser.add_argument(
        "--from-stage",
        type=str,
        default="30",
        help="Stage to start execution from (30, 40). Default: 30",
    )
    remaining_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Run profile: research (default) or smoke",
    )
    remaining_parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Protocol split index ('0', '1', '2', or 'all'). Default: all",
    )
    remaining_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    remaining_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values: --set key=value",
    )
    remaining_parser.add_argument(
        "--models",
        "--model",
        nargs="+",
        dest="models",
        default=None,
        help="Filter trials by model backbone(s)",
    )
    remaining_parser.add_argument(
        "--trials",
        "--trial",
        nargs="+",
        dest="trials",
        default=None,
        help="Filter trials by experiment ID(s)",
    )

    # --- cystods validate ---
    validate_parser = subparsers.add_parser(
        "validate", help="Validate config without running"
    )
    validate_parser.add_argument("--config", type=str, default=None)
    validate_parser.add_argument("--profile", type=str, default=None)
    validate_parser.add_argument("--stage", type=str, default=None)
    validate_parser.add_argument("--split", type=str, default=None)
    validate_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values: --set key=value",
    )

    # --- cystods migrate-results ---
    migrate_parser = subparsers.add_parser(
        "migrate-results", help="Migrate legacy result directories to unified structure"
    )
    migrate_parser.add_argument(
        "path",
        type=str,
        help="Path to legacy result run directory or result root folder",
    )

    return parser


def _cmd_stages() -> None:
    """List all available pipeline stages."""
    print("\nCystoDS Pipeline Stages")
    print("=" * 60)
    for stage_id, description in _STAGE_REGISTRY.items():
        print(f"  Stage {stage_id}  │  {description}")
    print()
    print("Usage: cystods run <stage> [--split 0|1|2] [--profile smoke|research]")
    print()


def _cmd_config_show(args: argparse.Namespace) -> None:
    """Display the resolved configuration."""
    from cystods.config import show_config

    overrides = list(args.overrides) if args.overrides else []
    if getattr(args, "split", None) is not None:
        overrides.append(f"protocol_split_index={args.split}")

    output = show_config(
        config_path=args.config,
        profile=args.profile,
        stage=args.stage,
        cli_overrides=overrides if overrides else None,
    )
    print(output)


def _cmd_validate(args: argparse.Namespace) -> None:
    """Validate config without running."""
    from cystods.config import load_config

    overrides = list(args.overrides) if args.overrides else []
    if getattr(args, "split", None) is not None:
        overrides.append(f"protocol_split_index={args.split}")

    try:
        config = load_config(
            config_path=args.config,
            profile=args.profile,
            stage=args.stage,
            cli_overrides=overrides if overrides else None,
        )
        print(f"✓ Config loaded successfully ({len(config)} keys)")
        print(f"  profile: {config.get('run_profile', 'research')}")
        if args.stage:
            print(f"  stage: {args.stage}")
        if config.get("protocol_split_index") is not None:
            print(f"  split: split_{config['protocol_split_index']}")
        print(f"  data_root: {config.get('data_root', 'N/A')}")
        print(f"  result_root: {config.get('result_root', 'N/A')}")
    except Exception as exc:
        print(f"✗ Config validation failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _parse_list_arg(values: list[str] | None) -> list[str] | None:
    """Parse list of strings or comma-separated strings into a clean list of unique terms."""
    if not values:
        return None
    result: list[str] = []
    for item in values:
        for part in item.split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
    return result if result else None


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a pipeline stage."""
    # Handle comma-separated list of stages (e.g. "30,35,36,40,90")
    if "," in str(args.stage):
        stage_list = [s.strip().zfill(2) for s in args.stage.split(",") if s.strip()]
        print(f"\n🚀 Running {len(stage_list)} stages in sequence: {', '.join(stage_list)}\n")
        for st in stage_list:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.stage = st
            _cmd_run(sub_args)
        return

    stage_padded = args.stage.zfill(2)

    if stage_padded not in _STAGE_REGISTRY and args.stage != "all":
        print(
            f"✗ Unknown stage: {args.stage}. "
            f"Available: {', '.join(sorted(_STAGE_REGISTRY))}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.stage == "all":
        _run_all(args)
        return

    from cystods.config import load_config

    filter_models = _parse_list_arg(getattr(args, "models", None))
    filter_trials = _parse_list_arg(getattr(args, "trials", None))

    split_str = str(getattr(args, "split", "all")).strip().lower()

    # Stages 35 and 36 natively handle multi-split iteration
    if stage_padded in ("35", "36"):
        config = load_config(
            config_path=args.config,
            profile=args.profile,
            stage=stage_padded,
            cli_overrides=args.overrides,
        )
        config["filter_models"] = filter_models
        config["filter_trials"] = filter_trials
        config["protocol_split_index"] = split_str

        print(f"\n{'='*60}")
        print(f"CystoDS — Stage {stage_padded}")
        print(f"  {_STAGE_REGISTRY[stage_padded]}")
        print(f"  Profile: {config['run_profile']}")
        print(f"  Split: {split_str}")
        print(f"{'='*60}\n")

        stage_module = _import_stage(stage_padded)
        result = stage_module.run(config)
        print(f"\n✓ Stage {stage_padded} completed: {result}\n")
        return

    # Stages 00 and 90 do not take split arguments
    if stage_padded in ("00", "90", "60"):
        config = load_config(
            config_path=args.config,
            profile=args.profile,
            stage=stage_padded,
            cli_overrides=args.overrides,
        )
        config["filter_models"] = filter_models
        config["filter_trials"] = filter_trials
        config["protocol_split_index"] = None

        print(f"\n{'='*60}")
        print(f"CystoDS — Stage {stage_padded}")
        print(f"  {_STAGE_REGISTRY[stage_padded]}")
        print(f"  Profile: {config['run_profile']}")
        print(f"{'='*60}\n")

        stage_module = _import_stage(stage_padded)
        result = stage_module.run(config)
        print(f"\n✓ Stage {stage_padded} completed: {result}\n")
        return

    # For standard stages (10, 20, 30, 40): resolve splits
    if split_str == "all":
        split_list = [0, 1, 2]
    else:
        split_list = [int(s.strip()) for s in split_str.split(",") if s.strip()]

    for s_idx in split_list:
        config = load_config(
            config_path=args.config,
            profile=args.profile,
            stage=stage_padded,
            cli_overrides=args.overrides,
        )
        config["filter_models"] = filter_models
        config["filter_trials"] = filter_trials
        config["protocol_split_index"] = s_idx
        if getattr(args, "freeze_stage3", False):
            config["partial_finetune"] = True
            config["freeze_early_layers"] = True
            config["frozen_stages_count"] = 3
        elif getattr(args, "freeze_stages", None) is not None:
            config["partial_finetune"] = True
            config["freeze_early_layers"] = True
            config["frozen_stages_count"] = int(args.freeze_stages)
        elif getattr(args, "partial_finetune", False):
            config["partial_finetune"] = True
            config["freeze_early_layers"] = True
            config["frozen_stages_count"] = config.get("frozen_stages_count", 2)

        print(f"\n{'='*60}")
        print(f"CystoDS — Stage {stage_padded} (Split {s_idx})")
        print(f"  {_STAGE_REGISTRY[stage_padded]}")
        print(f"  Profile: {config['run_profile']}")
        print(f"  Split: split_{s_idx}")
        if config.get("partial_finetune"):
            n_frz = config.get("frozen_stages_count", 2)
            print(f"  Finetune Mode: Partial (Frozen: PatchEmbed + Stages 1..{n_frz} | Trainable: Stages {n_frz+1}..4 + Heads)")
        if filter_models:
            print(f"  Filter Models: {', '.join(filter_models)}")
        if filter_trials:
            print(f"  Filter Trials: {', '.join(filter_trials)}")
        print(f"{'='*60}\n")

        stage_module = _import_stage(stage_padded)
        result = stage_module.run(config)
        print(f"\n✓ Stage {stage_padded} (Split {s_idx}) completed: {result}\n")


def _run_all(args: argparse.Namespace) -> None:
    """Run all stages in dependency order."""
    if args.split is None:
        print(
            "✗ Error: 'cystods run all' requires --split {0, 1, 2}.",
            file=sys.stderr,
        )
        sys.exit(1)

    order = ["00", "10", "20", "30", "40", "90"]

    filter_models = _parse_list_arg(getattr(args, "models", None))
    filter_trials = _parse_list_arg(getattr(args, "trials", None))

    from cystods.config import load_config

    for stage_id in order:
        config = load_config(
            config_path=args.config,
            profile=args.profile,
            stage=stage_id,
            cli_overrides=args.overrides,
        )
        config["filter_models"] = filter_models
        config["filter_trials"] = filter_trials
        if stage_id != "00":
            config["protocol_split_index"] = args.split
        else:
            config["protocol_split_index"] = None

        print(f"\n{'='*60}")
        print(f"CystoDS — Stage {stage_id}: {_STAGE_REGISTRY[stage_id]}")
        if config.get("protocol_split_index") is not None:
            print(f"  Split: split_{config['protocol_split_index']}")
        if filter_models:
            print(f"  Filter Models: {', '.join(filter_models)}")
        if filter_trials:
            print(f"  Filter Trials: {', '.join(filter_trials)}")
        print(f"{'='*60}\n")

        stage_module = _import_stage(stage_id)
        result = stage_module.run(config)
        print(f"✓ Stage {stage_id} completed: {result}\n")

    print("\n✓ All stages completed successfully.\n")


def _import_stage(stage_id: str):
    """Dynamically import a stage module."""
    import importlib

    module_name = f"cystods.stages.stage_{stage_id}"
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        print(f"✗ Failed to import {module_name}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_run_remaining(args: argparse.Namespace) -> None:
    """Run all remaining stages (30 -> 40) sequentially."""
    all_stages = ["30", "40"]
    start_stage = str(args.from_stage).zfill(2)
    if start_stage not in all_stages:
        print(
            f"✗ Unknown start stage: {args.from_stage}. Available: {', '.join(all_stages)}",
            file=sys.stderr,
        )
        sys.exit(1)

    idx = all_stages.index(start_stage)
    stages_to_run = all_stages[idx:]

    print("\n" + "=" * 70)
    print(f"🚀 CYSTODS PIPELINE: CHẠY CÁC GIAI ĐOẠN CÒN LẠI ({' -> '.join(stages_to_run)})")
    print(f"  • Profile  : {args.profile or 'research'}")
    print(f"  • Split    : {args.split}")
    print(f"  • Start at : Stage {start_stage}")
    print("=" * 70 + "\n")

    for st in stages_to_run:
        sub_args = argparse.Namespace(**vars(args))
        sub_args.stage = st
        if st in ("00", "90"):
            sub_args.split = None
        _cmd_run(sub_args)

    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH TOÀN BỘ CÁC GIAI ĐOẠN CÒN LẠI CỦA DỰ ÁN CYSTODS!")
    print("=" * 70 + "\n")


def _cmd_migrate_results(args: argparse.Namespace) -> None:
    """Migrate legacy result directories to the unified structure."""
    from cystods.migrate import migrate_result_directory

    target = Path(args.path).resolve()
    if target.name in ("result", "results"):
        for child in sorted(target.iterdir()):
            if child.is_dir() and not child.name.endswith("__runs") and not child.name.startswith("."):
                try:
                    res = migrate_result_directory(child)
                    print(f"✓ Migrated {child.name} -> {res}")
                except Exception as exc:
                    print(f"✗ Failed migrating {child.name}: {exc}", file=sys.stderr)
    else:
        try:
            res = migrate_result_directory(target)
            print(f"✓ Migrated {target.name} -> {res}")
        except Exception as exc:
            print(f"✗ Failed migrating {target.name}: {exc}", file=sys.stderr)
            sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "stages":
        _cmd_stages()
    elif args.command == "config":
        if getattr(args, "config_command", None) == "show":
            _cmd_config_show(args)
        else:
            print("Usage: cystods config show [--stage S]", file=sys.stderr)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "run-remaining":
        _cmd_run_remaining(args)
    elif args.command == "migrate-results":
        _cmd_migrate_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
