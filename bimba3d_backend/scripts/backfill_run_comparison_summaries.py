from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def build_summary(project_id: str, run_id: str, run_dir: Path, engine: str = "gsplat") -> dict[str, Any]:
    engine_dir = run_dir / "outputs" / "engines" / engine
    eval_path = engine_dir / "eval_history.json"
    metadata_path = engine_dir / "metadata.json"
    tuning_path = engine_dir / "adaptive_tuning_results.json"
    run_config_path = run_dir / "run_config.json"

    eval_history_raw = _read_json(eval_path)
    if not isinstance(eval_history_raw, list):
        eval_history_raw = []
    eval_history = sorted(
        [item for item in eval_history_raw if isinstance(item, dict)],
        key=lambda item: item.get("step") if isinstance(item.get("step"), (int, float)) else float("inf"),
    )

    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        metadata = {}

    run_config = _read_json(run_config_path)
    if not isinstance(run_config, dict):
        run_config = {}

    tuning_results: dict[str, Any] = {}
    if tuning_path.exists():
        parsed = _read_json(tuning_path)
        if isinstance(parsed, dict):
            tuning_results = parsed

    latest_eval = eval_history[-1] if eval_history else {}
    first_eval = eval_history[0] if eval_history else {}

    eval_series: list[dict[str, float | int]] = []
    eval_time_series: list[dict[str, float | int]] = []
    loss_milestones: dict[str, float] = {}

    for point in eval_history:
        step = point.get("step")
        if isinstance(step, (int, float)):
            loss_value = point.get("final_loss")
            if isinstance(loss_value, (int, float)):
                eval_series.append({"step": int(step), "loss": float(loss_value)})

            elapsed_seconds = point.get("elapsed_seconds")
            if isinstance(elapsed_seconds, (int, float)) and elapsed_seconds >= 0:
                eval_time_series.append({
                    "step": int(step),
                    "elapsed_seconds": float(elapsed_seconds),
                })

        for key, value in point.items():
            if isinstance(key, str) and key.startswith("loss_at_") and isinstance(value, (int, float)):
                loss_milestones[key] = float(value)

    total_time_seconds = None
    if eval_time_series:
        total_time_seconds = max(float(point["elapsed_seconds"]) for point in eval_time_series)

    resolved = run_config.get("resolved_params")
    if not isinstance(resolved, dict):
        resolved = {}

    tuning_history = tuning_results.get("tuning_history")
    if not isinstance(tuning_history, list):
        tuning_history = []

    runtime_series = [
        {"step": item.get("step"), "params": item.get("params")}
        for item in tuning_history
        if isinstance(item, dict) and isinstance(item.get("step"), (int, float)) and isinstance(item.get("params"), dict)
    ]

    final_loss = latest_eval.get("final_loss") if isinstance(latest_eval, dict) else None
    if not isinstance(final_loss, (int, float)):
        final_loss = None

    best_splat = metadata.get("best_splat") if isinstance(metadata.get("best_splat"), dict) else {}
    best_splat_step = best_splat.get("step") if isinstance(best_splat.get("step"), (int, float)) else None
    best_splat_loss = best_splat.get("loss") if isinstance(best_splat.get("loss"), (int, float)) else None

    return {
        "project_id": project_id,
        "run_id": run_id,
        "run_name": run_config.get("run_name") or run_id,
        "name": project_id,
        "status": "completed",
        "mode": metadata.get("mode"),
        "engine": engine,
        "metrics": {
            "convergence_speed": latest_eval.get("convergence_speed") if isinstance(latest_eval, dict) else None,
            "final_loss": final_loss,
            "lpips_mean": latest_eval.get("lpips_mean") if isinstance(latest_eval, dict) else None,
            "sharpness_mean": latest_eval.get("sharpness_mean") if isinstance(latest_eval, dict) else None,
            "num_gaussians": latest_eval.get("num_gaussians") if isinstance(latest_eval, dict) else None,
            "total_time_seconds": total_time_seconds,
            "best_splat_step": int(best_splat_step) if isinstance(best_splat_step, (int, float)) else None,
            "best_splat_loss": float(best_splat_loss) if isinstance(best_splat_loss, (int, float)) else None,
        },
        "tuning": {
            "initial": first_eval.get("tuning_params") if isinstance(first_eval, dict) and isinstance(first_eval.get("tuning_params"), dict) else {},
            "final": latest_eval.get("tuning_params") if isinstance(latest_eval, dict) and isinstance(latest_eval.get("tuning_params"), dict) else {},
            "end_params": tuning_results.get("final_params") if isinstance(tuning_results.get("final_params"), dict) else {},
            "end_step": resolved.get("tune_end_step") if resolved.get("tune_end_step") is not None else tuning_results.get("tune_end_step"),
            "runs": metadata.get("tuning_runs"),
            "history_count": len(tuning_history),
            "history": tuning_history,
            "tune_interval": resolved.get("tune_interval"),
            "log_interval": resolved.get("log_interval"),
            "runtime_series": runtime_series,
        },
        "major_params": {
            "max_steps": resolved.get("max_steps"),
            "total_steps_completed": latest_eval.get("step") if isinstance(latest_eval, dict) else None,
            "densify_from_iter": resolved.get("densify_from_iter"),
            "densify_until_iter": resolved.get("densify_until_iter"),
            "densification_interval": resolved.get("densification_interval"),
            "eval_interval": resolved.get("eval_interval"),
            "save_interval": resolved.get("save_interval"),
            "splat_export_interval": resolved.get("splat_export_interval"),
            "batch_size": resolved.get("batch_size"),
        },
        "loss_milestones": loss_milestones,
        "eval_series": eval_series,
        "eval_time_series": eval_time_series,
        "preview_url": None,
        "eval_points": len(eval_history),
    }


def backfill_runs(data_projects_dir: Path, project_filter: str | None = None, dry_run: bool = False) -> tuple[int, int]:
    updated = 0
    skipped = 0

    project_dirs = [p for p in data_projects_dir.iterdir() if p.is_dir()]
    for project_dir in sorted(project_dirs):
        project_id = project_dir.name
        if project_filter and project_filter != project_id:
            continue

        runs_dir = project_dir / "runs"
        if not runs_dir.exists():
            continue

        for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
            run_id = run_dir.name
            engine_dir = run_dir / "outputs" / "engines" / "gsplat"
            required = [
                run_dir / "run_config.json",
                engine_dir / "eval_history.json",
                engine_dir / "metadata.json",
            ]

            if not all(path.exists() for path in required):
                skipped += 1
                continue

            comparison_dir = run_dir / "comparison"
            summary_path = comparison_dir / "experiment_summary.json"

            summary_payload = build_summary(project_id, run_id, run_dir, engine="gsplat")
            if dry_run:
                print(f"[DRY-RUN] would write {summary_path}")
                updated += 1
                continue

            _write_json(summary_path, summary_payload)

            for name in ("eval_history.json", "adaptive_tuning_results.json", "metadata.json"):
                source = engine_dir / name
                if source.exists():
                    comparison_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, comparison_dir / name)

            run_cfg = run_dir / "run_config.json"
            if run_cfg.exists():
                comparison_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_cfg, comparison_dir / "run_config.json")

            updated += 1

    return updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill run comparison summaries for existing gsplat runs.")
    parser.add_argument("--project-id", help="Optional single project id to backfill.")
    parser.add_argument("--data-projects-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "projects"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_projects_dir = Path(args.data_projects_dir)
    if not data_projects_dir.exists():
        raise SystemExit(f"Data projects dir not found: {data_projects_dir}")

    updated, skipped = backfill_runs(data_projects_dir, project_filter=args.project_id, dry_run=args.dry_run)
    print(f"Backfill complete. updated={updated} skipped={skipped} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
