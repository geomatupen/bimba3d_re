#!/usr/bin/env python3
"""Batch export per-run PDFs (telemetry, AI log, comparison-vs-base) and a 3-column link table.

This script uses backend APIs that power the UI tabs:
- Process full log telemetry: GET /projects/{project_id}/telemetry
- AI logs (derived from telemetry event_rows)
- Comparison data: GET /projects/{project_id}/experiment-summary

Output:
- 3 PDFs per run
- markdown table with run_name, pdf_type, and PDF filename link text
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def api(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
    url = f"{base_url}{path}"
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}?{query}"
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=240) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def sanitize_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-_") or "item"


def fmt_num(value: Any, ndigits: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{ndigits}f}"
    return "-"


def build_key_value_rows(items: list[tuple[str, Any]]) -> list[list[str]]:
    rows = [["Field", "Value"]]
    for key, value in items:
        if isinstance(value, bool):
            display = "Yes" if value else "No"
        elif value is None:
            display = "-"
        else:
            display = str(value)
        rows.append([key, display])
    return rows


def build_pdf(path: Path, title: str, sections: list[tuple[str, list[list[str]]]]) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm)
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}", styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    for section_title, rows in sections:
        story.append(Paragraph(section_title, styles["Heading3"]))
        table = Table(rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#193778")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 4 * mm))

    doc.build(story)


def extract_ai_events(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    events = telemetry.get("event_rows") if isinstance(telemetry, dict) else None
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for row in events:
        if not isinstance(row, dict):
            continue
        row_type = str(row.get("type") or "").lower()
        if "ai" not in row_type:
            continue
        out.append(row)
    return out


def export_for_run(
    base_url: str,
    project_id: str,
    run_id: str,
    base_project_id: str,
    base_run_id: str,
    out_dir: Path,
    telemetry_only: bool = False,
) -> dict[str, str]:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_token = sanitize_token(run_id)
    proj_token = sanitize_token(project_id)

    telemetry = api(
        base_url,
        "GET",
        f"/projects/{project_id}/telemetry",
        params={"run_id": run_id, "log_limit": 500, "eval_limit": 100, "from_start": 1},
    )
    run_config_payload = api(base_url, "GET", f"/projects/{project_id}/runs/{run_id}/config")
    run_config = run_config_payload.get("run_config") if isinstance(run_config_payload, dict) else {}
    resolved_params = run_config.get("resolved_params") if isinstance(run_config, dict) else {}
    requested_params = run_config.get("requested_params") if isinstance(run_config, dict) else {}
    if not isinstance(resolved_params, dict):
        resolved_params = {}
    if not isinstance(requested_params, dict):
        requested_params = {}

    train_rows = telemetry.get("training_rows") if isinstance(telemetry.get("training_rows"), list) else []
    eval_rows = telemetry.get("eval_rows") if isinstance(telemetry.get("eval_rows"), list) else []
    event_rows = telemetry.get("event_rows") if isinstance(telemetry.get("event_rows"), list) else []

    telemetry_pdf = f"{proj_token}_{run_token}_telemetry_{stamp}.pdf"
    telemetry_path = out_dir / telemetry_pdf
    telemetry_sections: list[tuple[str, list[list[str]]]] = [
        (
            "Session Summary",
            [
                ["Field", "Value"],
                ["Project ID", project_id],
                ["Run ID", run_id],
                ["Training rows", str(len(train_rows))],
                ["Eval rows", str(len(eval_rows))],
                ["Event rows", str(len(event_rows))],
                ["Current stage", str((telemetry.get("status") or {}).get("stage") or "-")],
                ["Current step", str((telemetry.get("status") or {}).get("currentStep") or "-")],
                ["Current loss", fmt_num((telemetry.get("status") or {}).get("current_loss"), 8)],
            ],
        )
    ]

    resolved_config_rows = build_key_value_rows(
        [
            ("mode", resolved_params.get("mode")),
            ("engine", resolved_params.get("engine")),
            ("stage", resolved_params.get("stage")),
            ("tune_scope", resolved_params.get("tune_scope")),
            ("trend_scope", resolved_params.get("trend_scope")),
            ("max_steps", resolved_params.get("max_steps")),
            ("log_interval", resolved_params.get("log_interval")),
            ("eval_interval", resolved_params.get("eval_interval")),
            ("save_interval", resolved_params.get("save_interval")),
            ("splat_export_interval", resolved_params.get("splat_export_interval")),
            ("best_splat_interval", resolved_params.get("best_splat_interval")),
            ("best_splat_start_step", resolved_params.get("best_splat_start_step")),
            ("densify_from_iter", resolved_params.get("densify_from_iter")),
            ("densify_until_iter", resolved_params.get("densify_until_iter")),
            ("densification_interval", resolved_params.get("densification_interval")),
            ("batch_size", resolved_params.get("batch_size")),
            ("tune_start_step", resolved_params.get("tune_start_step")),
            ("tune_end_step", resolved_params.get("tune_end_step")),
            ("tune_interval", resolved_params.get("tune_interval")),
            ("tune_min_improvement", resolved_params.get("tune_min_improvement")),
            ("ai_lr_up_multiplier", resolved_params.get("ai_lr_up_multiplier")),
            ("ai_lr_down_multiplier", resolved_params.get("ai_lr_down_multiplier")),
            ("ai_gate_alpha", resolved_params.get("ai_gate_alpha")),
            ("ai_cooldown_intervals", resolved_params.get("ai_cooldown_intervals")),
            ("ai_small_change_band", resolved_params.get("ai_small_change_band")),
            ("ai_reward_step_weight", resolved_params.get("ai_reward_step_weight")),
            ("ai_reward_trend_weight", resolved_params.get("ai_reward_trend_weight")),
        ]
    )
    telemetry_sections.append(("Run Configuration (resolved)", resolved_config_rows))

    requested_config_rows = build_key_value_rows(
        [
            ("mode", requested_params.get("mode")),
            ("engine", requested_params.get("engine")),
            ("stage", requested_params.get("stage")),
            ("trend_scope", requested_params.get("trend_scope")),
            ("tune_interval", requested_params.get("tune_interval")),
        ]
    )
    telemetry_sections.append(("Run Configuration (requested)", requested_config_rows))

    latest_train = list(train_rows[-40:])
    if latest_train:
        rows = [["Timestamp", "Step", "Loss", "Elapsed(s)", "ETA", "Speed"]]
        for r in latest_train:
            if not isinstance(r, dict):
                continue
            rows.append(
                [
                    str(r.get("timestamp") or "-")[:19],
                    str(r.get("step") or "-"),
                    fmt_num(r.get("loss"), 8),
                    fmt_num(r.get("elapsed_seconds"), 2),
                    str(r.get("eta") or "-"),
                    str(r.get("speed") or "-"),
                ]
            )
        telemetry_sections.append(("Training Log (tail)", rows))

    build_pdf(
        telemetry_path,
        f"Telemetry Full Log | {project_id} | {run_id}",
        telemetry_sections,
    )

    if telemetry_only:
        return {"telemetry": telemetry_pdf}

    ai_events = extract_ai_events(telemetry)
    ai_pdf = f"{proj_token}_{run_token}_ai_logs_{stamp}.pdf"
    ai_path = out_dir / ai_pdf
    ai_rows = [["Step", "Action", "Reason", "Rel.Impr", "Reward", "Type"]]
    for e in ai_events:
        ai_rows.append(
            [
                str(e.get("step") or "-"),
                str(e.get("action") or "-"),
                str(e.get("reason") or e.get("summary") or "-")[:120],
                fmt_num(e.get("relative_improvement"), 6),
                fmt_num(e.get("reward_from_previous"), 6),
                str(e.get("type") or "-"),
            ]
        )
    if len(ai_rows) == 1:
        ai_rows.append(["-", "-", "No AI log events found", "-", "-", "-"])

    left_summary = api(base_url, "GET", f"/projects/{base_project_id}/experiment-summary", params={"run_id": base_run_id})
    right_summary = api(base_url, "GET", f"/projects/{project_id}/experiment-summary", params={"run_id": run_id})

    comp_pdf = f"comparison_{sanitize_token(base_project_id)}_{sanitize_token(base_run_id)}_vs_{proj_token}_{run_token}_{stamp}.pdf"
    comp_path = out_dir / comp_pdf

    metric_keys = ["best_loss", "final_eval_loss", "best_psnr", "best_lpips", "best_ssim", "total_elapsed_seconds"]
    metric_rows = [["Metric", "Base", "Run", "Delta (Run-Base)"]]
    left_metrics = left_summary.get("metrics") if isinstance(left_summary.get("metrics"), dict) else {}
    right_metrics = right_summary.get("metrics") if isinstance(right_summary.get("metrics"), dict) else {}
    for key in metric_keys:
        lv = left_metrics.get(key)
        rv = right_metrics.get(key)
        delta = "-"
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            delta = f"{float(rv) - float(lv):.6f}"
        metric_rows.append([key, fmt_num(lv), fmt_num(rv), delta])

    major_keys = [
        "max_steps",
        "densify_from_iter",
        "densify_until_iter",
        "densification_interval",
        "eval_interval",
        "best_splat_interval",
        "auto_early_stop",
        "batch_size",
    ]
    major_rows = [["Param", "Base", "Run"]]
    left_major = left_summary.get("major_params") if isinstance(left_summary.get("major_params"), dict) else {}
    right_major = right_summary.get("major_params") if isinstance(right_summary.get("major_params"), dict) else {}
    for key in major_keys:
        major_rows.append([key, str(left_major.get(key, "-")), str(right_major.get(key, "-"))])
    build_pdf(
        ai_path,
        f"AI Logs | {project_id} | {run_id}",
        [("AI Events", ai_rows)],
    )
    build_pdf(
        comp_path,
        f"Comparison vs Base | base={base_project_id}/{base_run_id} | run={project_id}/{run_id}",
        [("Metrics", metric_rows), ("Major Params", major_rows)],
    )

    return {
        "telemetry": telemetry_pdf,
        "ai": ai_pdf,
        "comparison": comp_pdf,
    }


def detect_default_project(base_url: str) -> str:
    # Prefer the known mx12 project when present, otherwise fallback to first project with completed runs.
    projects = api(base_url, "GET", "/projects/")
    if not isinstance(projects, list):
        raise RuntimeError("Failed to load projects list")

    def completed_runs(pid: str) -> list[dict[str, Any]]:
        payload = api(base_url, "GET", f"/projects/{pid}/runs")
        runs = payload.get("runs") if isinstance(payload, dict) else []
        if not isinstance(runs, list):
            return []
        return [r for r in runs if isinstance(r, dict) and r.get("session_status") == "completed"]

    candidates: list[tuple[str, int, int]] = []
    for p in projects:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("project_id") or "")
        if not pid:
            continue
        runs = completed_runs(pid)
        mx12_count = sum(1 for r in runs if str(r.get("run_id") or "").startswith("mx12_"))
        candidates.append((pid, mx12_count, len(runs)))

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if not candidates:
        raise RuntimeError("No projects found")
    return candidates[0][0]


def resolve_base_run_id(base_url: str, base_project_id: str, preferred_run_id: str) -> str:
    runs_payload = api(base_url, "GET", f"/projects/{base_project_id}/runs")
    runs = runs_payload.get("runs") if isinstance(runs_payload, dict) else []
    if not isinstance(runs, list):
        return preferred_run_id

    def has_summary(run_id: str) -> bool:
        if not run_id:
            return False
        try:
            api(base_url, "GET", f"/projects/{base_project_id}/experiment-summary", params={"run_id": run_id})
            return True
        except Exception:
            return False

    run_ids = [str(r.get("run_id") or "") for r in runs if isinstance(r, dict)]
    if preferred_run_id in run_ids and has_summary(preferred_run_id):
        return preferred_run_id

    base_candidates = [
        str(r.get("run_id") or "")
        for r in runs
        if isinstance(r, dict) and r.get("is_base") and r.get("session_status") == "completed"
    ]
    for run_id in base_candidates:
        if has_summary(run_id):
            return run_id

    completed = [
        str(r.get("run_id") or "")
        for r in runs
        if isinstance(r, dict) and r.get("session_status") == "completed"
    ]
    for run_id in completed:
        if has_summary(run_id):
            return run_id

    return preferred_run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch export telemetry/AI/comparison PDFs per run.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006")
    parser.add_argument("--project-id", default="auto", help="Target project to export. Use 'auto' to detect.")
    parser.add_argument("--base-project-id", default="7a10c332-ce34-4b0f-bc79-da8c7385d97f")
    parser.add_argument("--base-run-id", default="base")
    parser.add_argument("--include-base", action="store_true", help="Include the base run in exports")
    parser.add_argument("--out-root", default="bimba3d_backend/data/exports")
    parser.add_argument("--telemetry-only", action="store_true", help="Export only the full telemetry PDF for each run")
    args = parser.parse_args()

    try:
        project_id = args.project_id
        if project_id == "auto":
            project_id = detect_default_project(args.base_url)

        resolved_base_run_id = resolve_base_run_id(args.base_url, args.base_project_id, args.base_run_id)

        runs_payload = api(args.base_url, "GET", f"/projects/{project_id}/runs")
        runs = runs_payload.get("runs") if isinstance(runs_payload, dict) else []
        if not isinstance(runs, list):
            raise RuntimeError("Unexpected runs payload")

        completed = [r for r in runs if isinstance(r, dict) and r.get("session_status") == "completed"]
        completed.sort(key=lambda r: str(r.get("run_id") or ""))

        if not args.include_base:
            completed = [r for r in completed if not bool(r.get("is_base")) and str(r.get("run_id") or "") != "base"]

        if not completed:
            raise RuntimeError("No completed runs to export")

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_root) / f"pdf_batch_{project_id}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        for run in completed:
            run_id = str(run.get("run_id") or "").strip()
            if not run_id:
                continue
            try:
                files = export_for_run(
                    base_url=args.base_url,
                    project_id=project_id,
                    run_id=run_id,
                    base_project_id=args.base_project_id,
                    base_run_id=resolved_base_run_id,
                    out_dir=out_dir,
                    telemetry_only=args.telemetry_only,
                )
                rows.append({"run_name": run_id, **files})
                print(f"OK {run_id}")
            except urllib.error.HTTPError as e:
                print(f"FAIL {run_id}: HTTP {e.code}")
            except Exception as e:
                print(f"FAIL {run_id}: {e}")

        table_md = out_dir / "pdf_exports_table.md"
        lines = [
            "| run_name | PDF Type | PDF File |",
            "|---|---|---|",
        ]
        for row in rows:
            t = row["telemetry"]
            lines.append(f"| {row['run_name']} | Telemetry | [{t}]({t}) |")
            if not args.telemetry_only:
                a = row["ai"]
                c = row["comparison"]
                lines.append(f"| {row['run_name']} | AI log | [{a}]({a}) |")
                lines.append(f"| {row['run_name']} | Comparison | [{c}]({c}) |")
        table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

        manifest = out_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "project_id": project_id,
                    "base_project_id": args.base_project_id,
                    "base_run_id": resolved_base_run_id,
                    "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "rows": rows,
                    "table": table_md.name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"DONE project={project_id}")
        print(f"OUT {out_dir}")
        print(f"TABLE {table_md}")
        print(f"ROWS {len(rows)}")
        return 0
    except Exception as exc:
        print(f"ERROR {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
