from pathlib import Path

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PACKAGE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = PACKAGE_DIR / "static"
