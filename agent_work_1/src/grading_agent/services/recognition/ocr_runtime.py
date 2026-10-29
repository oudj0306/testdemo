"""One resident OCR model per device; serialize GPU inference to bound VRAM."""

import importlib.util
import json
import os
import re
import sys
from threading import RLock

_lock = RLock()
_engines = {}
_errors = {}
_initialization_errors = set()
_active_device = None
_compatibility_warning = None


def runtime_status():
    return {
        "python_executable": sys.executable,
        "model_cache": os.environ.get("PADDLE_PDX_CACHE_HOME"),
        "paddle_installed": importlib.util.find_spec("paddle") is not None,
        "paddleocr_installed": importlib.util.find_spec("paddleocr") is not None,
        "active_device": _active_device,
        "errors": dict(_errors),
        "compatibility_warning": _compatibility_warning,
    }


def recognize(path, device="auto", retry_initialization=False, leading_number=False):
    global _active_device, _compatibility_warning
    with _lock:
        try:
            if retry_initialization:
                _initialization_errors.discard(device)
            if device in _initialization_errors:
                return "", None, "paddleocr_failed"
            if device not in _engines:
                from grading_agent.services.runtime_paths import model_cache_root

                cache = model_cache_root()
                os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache / "paddlex"))
                os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
                import paddle
                from paddleocr import PaddleOCR

                gpu_available = (
                    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
                )
                selected = ("gpu:0" if gpu_available else "cpu") if device == "auto" else device
                if selected.startswith("gpu") and not gpu_available:
                    raise RuntimeError(
                        "请求 GPU，但当前 Python 没有可用的 Paddle GPU 环境；运行 setup-gpu.ps1"
                    )
                if selected.startswith("gpu") and hasattr(paddle, "version"):
                    compiled = tuple(int(p) for p in paddle.version.cudnn().split("."))
                    raw_version = paddle.device.get_cudnn_version()
                    runtime = (
                        (raw_version // 10000, raw_version % 10000 // 100, raw_version % 100)
                        if raw_version >= 90000
                        else (raw_version // 1000, raw_version % 1000 // 100, raw_version % 100)
                    )
                    if runtime < compiled:
                        _compatibility_warning = f"cuDNN 编译版本 {'.'.join(map(str, compiled))} 高于运行版本 {'.'.join(map(str, runtime))}；存在官方依赖兼容警告，请用真实样本验证稳定性。"
                # 'auto' and 'gpu:0' may resolve to the same resident model.
                engine = next((e for e, actual in _engines.values() if actual == selected), None)
                if engine is None:
                    engine = PaddleOCR(
                        device=selected,
                        text_detection_model_name="PP-OCRv5_mobile_det",
                        text_recognition_model_name="PP-OCRv5_server_rec",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        text_rec_score_thresh=0.0,
                        # Avoid the Windows Paddle PIR/oneDNN attribute conversion failure.
                        **({"enable_mkldnn": False} if selected == "cpu" else {}),
                    )
                _engines[device] = (engine, selected)
            engine, selected = _engines[device]
            texts, scores = [], []
            for result in engine.predict(str(path)):
                payload = getattr(result, "json", result)
                if isinstance(payload, str):
                    payload = json.loads(payload)
                data = payload.get("res", payload) if isinstance(payload, dict) else {}
                rec_texts, rec_scores = data.get("rec_texts", []), data.get("rec_scores", [])
                for index, text in enumerate(rec_texts):
                    if str(text).strip():
                        if leading_number and not re.match(r"^\s*\d{1,2}(?!\d)", str(text)):
                            continue
                        texts.append(str(text))
                        scores.append(float(rec_scores[index]) if index < len(rec_scores) else 0.0)
            if leading_number and texts:
                best = max(range(len(scores)), key=lambda i: scores[i])
                texts, scores = [texts[best]], [scores[best]]
            _active_device = selected
            _errors.pop(device, None)
            # A strong line must not hide an unreadable line by averaging scores.
            return "\n".join(texts), min(scores) if scores else None, f"paddleocr_{selected}"
        except Exception as exc:
            _errors[device] = str(exc)[:500]
            if device not in _engines:
                _initialization_errors.add(device)
            return "", None, "paddleocr_failed"
