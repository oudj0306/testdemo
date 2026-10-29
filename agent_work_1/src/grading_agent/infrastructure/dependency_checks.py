"""只读环境检查：区分缺包、版本不符和共享二进制模块冲突。"""

from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)
PADDLE_DISTRIBUTIONS = ("paddlepaddle", "paddlepaddle-gpu")
PROFILE_EXTRAS = {
    "basic": (),
    "local": ("local_recognition", "ocr_engine"),
    "cpu": ("local_recognition", "ocr"),
    "gpu": ("local_recognition", "ocr_engine"),
}


def check_environment(
    project: Mapping[str, Any],
    profile: str = "local",
    *,
    include_dev: bool = False,
    installed_version: Callable[[str], str] = version,
) -> dict[str, Any]:
    """检查已安装元数据；不安装包、不加载 GPU、不连接模型或读取业务配置。"""
    if profile not in PROFILE_EXTRAS:
        raise ValueError(f"Unknown dependency profile: {profile}")
    optional = project.get("optional-dependencies", {})
    requirements = list(project["dependencies"])
    for group in (*PROFILE_EXTRAS[profile], *(("dev",) if include_dev else ())):
        requirements.extend(optional[group])
    if profile == "gpu":
        requirements.append("paddlepaddle-gpu==3.3.0")
    elif profile == "local":
        # Prefer the installed GPU variant. Never demand the CPU wheel beside it.
        requirements.append(
            "paddlepaddle-gpu==3.3.0"
            if _installed("paddlepaddle-gpu", installed_version)
            else "paddlepaddle>=3.3,<3.4"
        )
    packages = []
    for text in dict.fromkeys(requirements):
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = _installed(requirement.name, installed_version)
        status = "missing" if installed is None else "ok"
        if installed is not None:
            try:
                if Version(installed) not in requirement.specifier:
                    status = "incompatible"
            except InvalidVersion:
                status = "invalid_version"
        packages.append({"requirement": text, "installed": installed, "status": status})
    conflicts = _conflicts(OPENCV_DISTRIBUTIONS, installed_version)
    if profile != "basic":
        conflicts.extend(_conflicts(PADDLE_DISTRIBUTIONS, installed_version))
    return {
        "ok": all(row["status"] == "ok" for row in packages) and not conflicts,
        "profile": profile,
        "packages": packages,
        "conflicts": conflicts,
    }


def _installed(name: str, installed_version: Callable[[str], str]) -> str | None:
    try:
        return installed_version(name)
    except PackageNotFoundError:
        return None


def _conflicts(names: Iterable[str], installed_version: Callable[[str], str]) -> list[list[str]]:
    installed = [name for name in names if _installed(name, installed_version) is not None]
    return [installed] if len(installed) > 1 else []
