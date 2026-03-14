import os

VALID_WORKER_MODES = {"docker", "local"}


def normalize_worker_mode(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip().lower()
    if not candidate:
        return None
    if candidate not in VALID_WORKER_MODES:
        raise ValueError(f"Invalid worker_mode '{value}'. Expected one of: docker, local")
    return candidate


def resolve_worker_mode(preferred: str | None = None) -> str:
    explicit = normalize_worker_mode(preferred)
    if explicit:
        return explicit

    env_mode = normalize_worker_mode(os.getenv("WORKER_MODE"))
    if env_mode:
        return env_mode

    legacy = os.getenv("USE_DOCKER_WORKER")
    if legacy is not None:
        return "docker" if legacy.strip().lower() == "true" else "local"

    return "docker"
