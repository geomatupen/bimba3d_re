#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-$HOME/bimba3d-shutdown-evidence}"
HOURS_BACK="${2:-12}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/$TS"
mkdir -p "$OUT_DIR"

{
  echo "Bimba3D WSL evidence"
  echo "Collected at: $(date -Is)"
  echo "Host: $(hostname)"
  echo "Kernel: $(uname -a)"
} > "$OUT_DIR/summary.txt"

{
  echo "===== boot_id ====="
  cat /proc/sys/kernel/random/boot_id
  echo
  echo "===== uptime ====="
  cat /proc/uptime
  echo
  echo "===== last reboot ====="
  last reboot -n 20 || true
} > "$OUT_DIR/boot_and_uptime.txt"

{
  echo "===== journalctl --list-boots ====="
  journalctl --list-boots || true
} > "$OUT_DIR/list_boots.txt"

{
  echo "===== previous boot tail ====="
  journalctl -b -1 -n 250 --no-pager || true
} > "$OUT_DIR/previous_boot_tail.txt"

{
  echo "===== current boot critical errors ====="
  journalctl -p 0..3 -b --no-pager || true
} > "$OUT_DIR/current_boot_errors.txt"

{
  echo "===== dmesg crash/power related ====="
  dmesg -T | grep -Ei 'oom|out of memory|panic|fatal|thermal|xid|reset|killed process|watchdog|power' || true
} > "$OUT_DIR/dmesg_filtered.txt"

if command -v docker >/dev/null 2>&1; then
  {
    echo "===== docker ps -a ====="
    docker ps -a
    echo
    echo "===== docker events (last ${HOURS_BACK}h) ====="
    SINCE="$(date -u -d "-${HOURS_BACK} hour" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
    if [[ -n "${SINCE:-}" ]]; then
      docker events --since "$SINCE" --until "$(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
    else
      echo "docker events skipped: failed to compute --since timestamp"
    fi
  } > "$OUT_DIR/docker_state_and_events.txt"
fi

printf "Evidence written to: %s\n" "$OUT_DIR"
