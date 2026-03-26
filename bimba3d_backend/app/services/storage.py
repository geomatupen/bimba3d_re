from pathlib import Path
import uuid
import os
import stat
import subprocess

from bimba3d_backend.app.config import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

STORAGE_ROOTS_ENV = "BIMBA3D_PROJECT_ROOTS"


def _safe_chown_chmod(path: Path, uid: int, gid: int, mode: int):
    """Safely chown and chmod a path if the current process has permissions.

    This attempts to set ownership to the given uid/gid and apply the mode.
    Failures are ignored (logged by caller if desired) so creation doesn't error
    on systems where chown is not permitted.
    """
    try:
        os.chown(path, uid, gid)
    except Exception:
        # Ignore: may not have permission to chown in some environments
        pass
    try:
        path.chmod(mode)
    except Exception:
        pass


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def list_storage_roots() -> list[dict]:
    roots: list[Path] = [DATA_DIR.resolve()]
    raw = os.environ.get(STORAGE_ROOTS_ENV, "").strip()
    if raw:
        candidates = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
        for candidate in candidates:
            try:
                path = Path(candidate).expanduser().resolve()
            except Exception:
                continue
            if path not in roots:
                roots.append(path)

    entries: list[dict] = []
    for idx, root in enumerate(roots):
        entries.append(
            {
                "id": "default" if idx == 0 else f"root_{idx}",
                "path": str(root),
                "label": "Default projects folder" if idx == 0 else f"Configured root {idx}",
                "is_default": idx == 0,
                "writable": _is_writable(root),
            }
        )
    return entries


def resolve_storage_root(storage_root_id: str | None = None, storage_path: str | None = None) -> Path:
    custom = (storage_path or "").strip()
    if custom:
        candidate = Path(custom).expanduser()
        if not candidate.is_absolute():
            raise ValueError("Storage path must be absolute")
        return candidate.resolve()

    roots = list_storage_roots()
    selected_id = (storage_root_id or "default").strip() or "default"
    for entry in roots:
        if entry["id"] == selected_id:
            return Path(entry["path"])

    raise ValueError(f"Unknown storage root id: {selected_id}")


def _link_project(project_alias: Path, target_dir: Path):
    if project_alias.exists():
        raise FileExistsError(f"Project alias already exists: {project_alias}")

    if os.name == "nt":
        try:
            os.symlink(str(target_dir), str(project_alias), target_is_directory=True)
            return
        except Exception:
            pass

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(project_alias), str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return

    os.symlink(str(target_dir), str(project_alias), target_is_directory=True)


def create_project(base_dir: Path | None = None):
    project_id = str(uuid.uuid4())
    root_dir = (base_dir or DATA_DIR).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    project_dir = root_dir / project_id
    project_alias = DATA_DIR / project_id
    images_dir = project_dir / "images"
    outputs_dir = project_dir / "outputs"
    images_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if project_alias != project_dir:
        _link_project(project_alias, project_dir)

    # Ensure created directories have restrictive but writable permissions for the backend user.
    # Use owner rwx, group rwx, others none (0o770). Also set setgid on the project dir so
    # newly created files inherit the group when possible (0o2770).
    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    uid = get_uid() if callable(get_uid) else None
    gid = get_gid() if callable(get_gid) else None
    try:
        # project_dir might not exist if parent mkdir race; ensure exists
        project_dir.mkdir(parents=True, exist_ok=True)
        if uid is not None and gid is not None:
            _safe_chown_chmod(project_dir, uid, gid, 0o2770)
            _safe_chown_chmod(images_dir, uid, gid, 0o770)
            _safe_chown_chmod(outputs_dir, uid, gid, 0o770)
            if project_alias.exists() and project_alias != project_dir:
                _safe_chown_chmod(project_alias, uid, gid, 0o2770)
        else:
            # Windows and other non-POSIX environments don't expose uid/gid.
            # Apply mode best-effort without ownership changes.
            project_dir.chmod(0o770)
            images_dir.chmod(0o770)
            outputs_dir.chmod(0o770)
            if project_alias.exists() and project_alias != project_dir:
                project_alias.chmod(0o770)
    except Exception:
        # Best-effort; do not fail project creation if permission setting isn't allowed
        pass

    return project_id, project_alias


def get_project_dir(project_id: str) -> Path:
    return DATA_DIR / project_id
