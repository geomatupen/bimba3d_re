from pathlib import Path
import uuid
import os
import stat

from bimba3d_backend.app.config import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def create_project():
    project_id = str(uuid.uuid4())
    project_dir = DATA_DIR / project_id
    images_dir = project_dir / "images"
    outputs_dir = project_dir / "outputs"
    images_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Ensure created directories have restrictive but writable permissions for the backend user.
    # Use owner rwx, group rwx, others none (0o770). Also set setgid on the project dir so
    # newly created files inherit the group when possible (0o2770).
    uid = os.getuid()
    gid = os.getgid()
    try:
        # project_dir might not exist if parent mkdir race; ensure exists
        project_dir.mkdir(parents=True, exist_ok=True)
        _safe_chown_chmod(project_dir, uid, gid, 0o2770)
        _safe_chown_chmod(images_dir, uid, gid, 0o770)
        _safe_chown_chmod(outputs_dir, uid, gid, 0o770)
    except Exception:
        # Best-effort; do not fail project creation if permission setting isn't allowed
        pass

    return project_id, project_dir


def get_project_dir(project_id: str) -> Path:
    return DATA_DIR / project_id
