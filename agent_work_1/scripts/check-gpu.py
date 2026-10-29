"""Synthetic GPU smoke test; never uploads documents or calls a paid model API."""

import json
import os
from pathlib import Path
from time import perf_counter

root = Path(__file__).resolve().parents[1]
cache = root / ".cache" / "gpu-check"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(root / ".cache" / "paddlex"))
os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))

import paddle
from PIL import Image, ImageDraw, ImageFont

from grading_agent.services.recognition.ocr.runtime import recognize, runtime_status

assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
paddle.set_device("gpu:0")
assert float((paddle.ones([2, 2]) @ paddle.ones([2, 2]))[0, 0]) == 2.0
path = cache / "synthetic-ocr.png"
image = Image.new("RGB", (1200, 360), "white")
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 56)
ImageDraw.Draw(image).text((40, 100), "GPU OCR TEST 12345", font=font, fill="black")
image.save(path)
timings = []
for _ in range(2):
    start = perf_counter()
    text, confidence, source = recognize(path, "gpu:0")
    timings.append(round(perf_counter() - start, 3))
    if "12345" not in text or source != "paddleocr_gpu:0":
        raise RuntimeError(
            json.dumps({"text": text, "runtime": runtime_status()}, ensure_ascii=False)
        )
print(
    json.dumps(
        {
            "device": paddle.device.cuda.get_device_name(),
            "text": text,
            "confidence": confidence,
            "source": source,
            "cold_seconds": timings[0],
            "warm_seconds": timings[1],
            "compatibility_warning": runtime_status()["compatibility_warning"],
        },
        ensure_ascii=False,
    )
)
