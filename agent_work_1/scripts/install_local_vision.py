"""Fetch only CPU/base + Vulkan files from the official Windows portable ZIP.

Pinned official HTTPS asset; verify each decompressed ZIP member's CRC and size.
No system installation, registry changes, third-party mirrors or model execution.
"""

import concurrent.futures
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

VERSION = "0.40.9"
URL = f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-windows-amd64.zip"
SIZE = 1586281618


def fetch(start, size):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            f"bytes {start}-"
        ):
            raise RuntimeError("Official server did not honor range request")
        data = response.read(size + 1)
    if len(data) != size:
        raise RuntimeError("Incomplete official ZIP range")
    return data


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else SIZE - self.position, SIZE - self.position)
        if size <= 0:
            return b""
        data = fetch(self.position, size)
        self.position += len(data)
        return data


def install(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZip()) as archive:
        entries = [e for e in archive.infolist() if not e.is_dir() and "/cuda_" not in e.filename]

    def member(entry):
        path = (root / entry.filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("Unsafe ZIP path")
        if (
            path.is_file()
            and path.stat().st_size == entry.file_size
            and zlib.crc32(path.read_bytes()) == entry.CRC
        ):
            return
        header = fetch(entry.header_offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError("Invalid official ZIP member")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        offset = entry.header_offset + 30 + name_length + extra_length
        chunks = [
            (offset + i, min(4 * 1024 * 1024, entry.compress_size - i))
            for i in range(0, entry.compress_size, 4 * 1024 * 1024)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            compressed = b"".join(pool.map(lambda spec: fetch(*spec), chunks))
        data = (
            zlib.decompress(compressed, -15)
            if entry.compress_type == zipfile.ZIP_DEFLATED
            else compressed
        )
        if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
            raise RuntimeError("ZIP member verification failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print("Verified", entry.filename, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(member, entries))
    (root / "origin.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "url": URL,
                "backend": "vulkan",
                "validation": "official HTTPS ZIP with per-member CRC32 and size; not full archive SHA256",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    args = parser.parse_args()
    install(args.destination)
