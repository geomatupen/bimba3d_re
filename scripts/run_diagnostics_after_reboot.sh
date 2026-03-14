#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_SCRIPT="$SCRIPT_DIR/collect_wsl_evidence.sh"
WIN_SCRIPT="$SCRIPT_DIR/collect_shutdown_evidence.ps1"

HOURS_BACK=12
WSL_OUT="${WSL_EVIDENCE_OUT:-$HOME/bimba3d-shutdown-evidence}"
WIN_OUT="${WIN_EVIDENCE_OUT:-C:\\temp\\bimba3d-shutdown-evidence}"
SKIP_WINDOWS=0
NO_UAC=0

usage() {
  cat <<'USAGE'
Usage: run_diagnostics_after_reboot.sh [options]

Options:
  --hours N            Hours back for log collection (default: 12)
  --wsl-out PATH       Output root for WSL evidence (default: $HOME/bimba3d-shutdown-evidence)
  --win-out PATH       Output root for Windows evidence (default: C:\temp\bimba3d-shutdown-evidence)
  --skip-windows       Collect only WSL logs
  --no-uac             Run Windows collector directly (non-elevated)
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours)
      HOURS_BACK="${2:-}"
      shift 2
      ;;
    --wsl-out)
      WSL_OUT="${2:-}"
      shift 2
      ;;
    --win-out)
      WIN_OUT="${2:-}"
      shift 2
      ;;
    --skip-windows)
      SKIP_WINDOWS=1
      shift
      ;;
    --no-uac)
      NO_UAC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$WSL_SCRIPT" ]]; then
  echo "WSL collector is missing or not executable: $WSL_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$WIN_SCRIPT" ]]; then
  echo "Windows collector script missing: $WIN_SCRIPT" >&2
  exit 1
fi

echo "[1/2] Collecting WSL evidence..."
"$WSL_SCRIPT" "$WSL_OUT" "$HOURS_BACK"

if [[ "$SKIP_WINDOWS" -eq 1 ]]; then
  echo "[2/2] Skipped Windows collector (--skip-windows)."
  exit 0
fi

WIN_SCRIPT_WIN="$(wslpath -w "$WIN_SCRIPT")"
WIN_SCRIPT_STAGED='C:\temp\collect_shutdown_evidence.ps1'

echo "[2/2] Preparing Windows collector script..."
powershell.exe -NoProfile -Command "\$src = '$WIN_SCRIPT_WIN'; \$dst = '$WIN_SCRIPT_STAGED'; New-Item -ItemType Directory -Force -Path 'C:\temp' | Out-Null; Copy-Item -LiteralPath \$src -Destination \$dst -Force"

echo "[2/2] Launching Windows collector..."
if [[ "$NO_UAC" -eq 1 ]]; then
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$WIN_SCRIPT_STAGED" -OutputRoot "$WIN_OUT" -HoursBack "$HOURS_BACK"
  echo "Windows collector finished (non-elevated)."
else
  powershell.exe -NoProfile -Command "Start-Process PowerShell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','$WIN_SCRIPT_STAGED','-OutputRoot','$WIN_OUT','-HoursBack','$HOURS_BACK'"
  echo "Windows UAC prompt launched. Approve it to collect full logs."
  echo "If no UAC prompt appears, run this manually in Windows PowerShell (Run as Administrator):"
  echo "  powershell -NoProfile -ExecutionPolicy Bypass -File \"$WIN_SCRIPT_STAGED\" -OutputRoot \"$WIN_OUT\" -HoursBack \"$HOURS_BACK\""
fi
