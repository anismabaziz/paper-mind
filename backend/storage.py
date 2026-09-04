"""
    Storage seam for uploaded PDF bytes.

    The app only knows the interface below; swapping local disk for a bucket
    service means implementing the same five methods.
"""

from pathlib import Path

import config


class LocalStorage:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: bytes) -> None:
        path = self._path(filename)
        path.write_bytes(content)

    def open(self, filename: str) -> bytes:
        return self._path(filename).read_bytes()

    def exists(self, filename: str) -> bool:
        return self._path(filename).is_file()

    def delete(self, filename: str) -> None:
        self._path(filename).unlink(missing_ok=True)

    def list(self) -> list[dict]:
        items = []
        for path in sorted(self._root.iterdir()):
            if path.is_file():
                items.append({"name": path.name, "size": path.stat().st_size})
        return items

    def url(self, filename: str) -> str:
        return f"/storage/{filename}"

    def _path(self, filename: str) -> Path:
        # Reject traversal: the name must resolve inside the storage root.
        path = (self._root / filename).resolve()
        if path.parent != self._root.resolve():
            raise ValueError(f"Invalid storage filename: {filename!r}")
        return path


storage = LocalStorage(config.STORAGE_DIR)
