from pathlib import Path
import hashlib


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)

    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()