"""Local synthetic OCR warmup, with no access to student submissions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont

from grading_agent.services.recognition.ocr.runtime import recognize, runtime_status


def check_ocr(device: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ocr-check-", dir=cache_dir) as directory:
        path = Path(directory) / "synthetic.png"
        image = Image.new("RGB", (1200, 360), "white")
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 56)
        except OSError:
            font = ImageFont.load_default(size=56)
        ImageDraw.Draw(image).text((40, 100), "GPU OCR TEST 12345", font=font, fill="black")
        image.save(path)
        durations = []
        for index in range(2):
            start = perf_counter()
            text, confidence, source = recognize(path, device, retry_initialization=index == 0)
            durations.append(round(perf_counter() - start, 3))
            # Do not treat an installed package or an empty prediction as success.
            if source == "paddleocr_failed" or "12345" not in text:
                return {
                    "ok": False,
                    "message": "合成图片 OCR 自检未通过，请查看设备错误信息",
                    "runtime": runtime_status(),
                    "durations_seconds": durations,
                }
        return {
            "ok": True,
            "device": runtime_status()["active_device"],
            "text": text,
            "confidence": confidence,
            "durations_seconds": durations,
            "runtime": runtime_status(),
            "message": "仅验证本地 OCR 可运行；耗时和置信不代表真实手写试卷准确率。",
        }
