from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def save_upload_file(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Можно загружать только .jpg, .jpeg, .png, .webp")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid4().hex}{suffix}"
    path = UPLOAD_DIR / filename

    with path.open("wb") as buffer:
        buffer.write(file.file.read())

    return str(path)
