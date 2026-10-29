"""Local layout-preserving Word -> PDF conversion, cached by file content."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from grading_agent.services.documents.ingestion import serialized_document_io

WORD_MEDIA = {
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def renderer_status():
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    for folder in (
        "C:/Program Files/LibreOffice/program",
        "C:/Program Files (x86)/LibreOffice/program",
    ):
        if not executable and (Path(folder) / "soffice.exe").is_file():
            executable = str(Path(folder) / "soffice.exe")
    if executable:
        return {"available": True, "backend": "libreoffice", "executable": executable}
    if os.name == "nt" and shutil.which("powershell"):
        import winreg

        for progid in ("KWPS.Application", "Word.Application"):
            try:
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid + r"\CLSID"):
                    return {
                        "available": True,
                        "backend": "office",
                        "executable": shutil.which("powershell"),
                    }
            except OSError:
                pass
    return {
        "available": False,
        "backend": None,
        "message": "Word 原卷需本机 Word/WPS 或 LibreOffice 转换以保留分页；也可将原卷导出 PDF 后上传框选。",
    }


@serialized_document_io
def word_to_pdf(source: Path, data_dir: Path):
    cache = data_dir / "document-previews"
    digest = hashlib.sha256(b"word-layout-v1" + source.read_bytes()).hexdigest()
    target = cache / (digest + ".pdf")
    if target.is_file() and target.read_bytes()[:5] == b"%PDF-":
        return target
    status = renderer_status()
    if not status["available"]:
        raise ValueError(status["message"])
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="word-", dir=cache) as temporary:
        directory = Path(temporary)
        if status["backend"] == "libreoffice":
            output = directory / (source.stem + ".pdf")
            command = [
                status["executable"],
                "-env:UserInstallation=" + (directory / "profile").as_uri(),
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(directory),
                str(source.resolve()),
            ]
        else:
            output = directory / "preview.pdf"
            command = [
                status["executable"],
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(Path(__file__).with_name("render_word.ps1")),
                "-InputPath",
                str(source.resolve()),
                "-OutputPath",
                str(output.resolve()),
            ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                "Word 分页转换超时；请关闭该文件的弹窗后重试，或导出 PDF 再上传。"
            ) from exc
        if result.returncode or not output.is_file() or output.read_bytes()[:5] != b"%PDF-":
            raise ValueError(
                "本机 Word/WPS 转换未完成；请检查软件激活、文件密码或兼容性，或导出 PDF 后上传。"
            )
        import pypdfium2 as pdfium

        try:
            with pdfium.PdfDocument(output) as pdf:
                if not 1 <= len(pdf) <= 100:
                    raise ValueError("Word 转换后页数必须在 1 到 100 之间")
        except pdfium.PdfiumError as exc:
            raise ValueError("Word 转换生成的 PDF 无法读取，请导出 PDF 后重试。") from exc
        output.replace(target)
    return target
