import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional


def parse_version(text: Optional[str]) -> Optional[tuple[int, int]]:
    if not text:
        return None
    match = re.search(r"(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def version_ge(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] > b[0] or (a[0] == b[0] and a[1] >= b[1])


def version_le(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])


def detect_cuda_version(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        parsed = parse_version(explicit)
        if parsed:
            return f"{parsed[0]}.{parsed[1]}"

    cuda_version_env = os.environ.get("CUDA_VERSION")
    if cuda_version_env:
        parsed = parse_version(cuda_version_env)
        if parsed:
            return f"{parsed[0]}.{parsed[1]}"

    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r"release\s+(\d+\.\d+)", output)
        if match:
            return match.group(1)
    except Exception:
        return None

    return None


def resolve_profile(matrix: dict, cuda_version: Optional[str]) -> dict:
    defaults = matrix["defaults"]

    profile = {
        "detected_cuda": cuda_version or "none",
        "torch_track": defaults["torchTrack"],
        "torch_index_url": defaults["torchIndexUrl"],
        "torch_version": defaults["torchVersion"],
        "torchvision_version": defaults["torchvisionVersion"],
        "torchaudio_version": defaults["torchaudioVersion"],
        "torch_cpu_version": defaults["torchCpuVersion"],
        "torchvision_cpu_version": defaults["torchvisionCpuVersion"],
        "torchaudio_cpu_version": defaults["torchaudioCpuVersion"],
        "gsplat_version": matrix["gsplat"]["version"],
        "colmap_version": matrix["colmap"]["cuda"]["version"],
        "use_cpu_torch": True,
        "install_gsplat": False,
    }

    min_cuda = parse_version(defaults["cudaMin"])
    detected = parse_version(cuda_version) if cuda_version else None

    if detected and min_cuda and version_ge(detected, min_cuda):
        profile["use_cpu_torch"] = False
        for candidate in matrix.get("torchProfiles", []):
            cmin = parse_version(candidate.get("minCuda"))
            cmax = parse_version(candidate.get("maxCuda")) if candidate.get("maxCuda") else None
            if not cmin:
                continue
            if not version_ge(detected, cmin):
                continue
            if cmax and not version_le(detected, cmax):
                continue

            profile["torch_track"] = candidate["track"]
            profile["torch_index_url"] = candidate["indexUrl"]
            profile["torch_version"] = candidate["torchVersion"]
            profile["torchvision_version"] = candidate["torchvisionVersion"]
            profile["torchaudio_version"] = candidate["torchaudioVersion"]
            break

    supported_tracks = set(matrix.get("gsplat", {}).get("supportedTorchTracks", []))
    profile["install_gsplat"] = (not profile["use_cpu_torch"]) and (profile["torch_track"] in supported_tracks)

    return profile


def as_shell_exports(profile: dict) -> str:
    lines = []
    for key, value in profile.items():
        env_key = key.upper()
        if isinstance(value, bool):
            env_value = "1" if value else "0"
        else:
            env_value = str(value)
        lines.append(f'export {env_key}="{env_value}"')
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve Docker worker compatibility profile")
    parser.add_argument("--matrix", default="compatibility-matrix.json")
    parser.add_argument("--cuda-version", default=None)
    parser.add_argument("--format", choices=["json", "shell"], default="shell")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Compatibility matrix not found: {matrix_path}")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cuda_version = detect_cuda_version(args.cuda_version)
    profile = resolve_profile(matrix, cuda_version)

    if args.format == "json":
        print(json.dumps(profile, indent=2))
    else:
        print(as_shell_exports(profile))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
