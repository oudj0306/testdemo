"""安装检查应识别真实缺包/冲突，不能把 GPU 环境误报为缺少 CPU Paddle。"""

import ast
import importlib.util
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from grading_agent.infrastructure.dependency_checks import check_environment
from grading_agent.infrastructure.runtime_paths import project_roots

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def installed(packages):
    def lookup(name):
        if name not in packages:
            raise PackageNotFoundError(name)
        return packages[name]

    return lookup


def project():
    return {
        "dependencies": ["fastapi>=0.115,<1"],
        "optional-dependencies": {
            "local_recognition": ["numpy>=1.26,<2.4", "opencv-contrib-python==4.10.0.84"],
            "ocr_engine": ["paddleocr>=3.7,<3.8"],
            "ocr": ["paddleocr>=3.7,<3.8", "paddlepaddle>=3.3,<3.4"],
            "dev": ["pytest>=8.3,<10"],
        },
    }


def gpu_packages():
    return {
        "fastapi": "0.143",
        "numpy": "2.3.5",
        "opencv-contrib-python": "4.10.0.84",
        "paddleocr": "3.7.0",
        "paddlepaddle-gpu": "3.3.0",
    }


@pytest.mark.parametrize("profile", ["local", "gpu"])
def test_gpu_environment_does_not_require_cpu_paddle(profile):
    result = check_environment(project(), profile, installed_version=installed(gpu_packages()))
    assert result["ok"]
    assert all(not row["requirement"].startswith("paddlepaddle>=") for row in result["packages"])


def test_missing_and_incompatible_dependencies_are_distinct():
    result = check_environment(
        project(), "basic", include_dev=True, installed_version=installed({"fastapi": "0.1"})
    )
    assert not result["ok"]
    assert [row["status"] for row in result["packages"]] == ["incompatible", "missing"]


@pytest.mark.parametrize(
    "extra", [{"paddlepaddle": "3.3.0"}, {"opencv-python-headless": "4.10.0.84"}]
)
def test_shared_binary_namespace_conflicts_are_reported(extra):
    result = check_environment(
        project(), "gpu", installed_version=installed({**gpu_packages(), **extra})
    )
    assert not result["ok"]
    assert len(result["conflicts"]) == 1


def test_launchers_reference_existing_python_modules():
    # Inline Python inside PowerShell is outside normal Python lint coverage.
    # This guard catches stale paths after moving OCR and storage modules.
    for path in [*PROJECT_ROOT.glob("*.ps1"), *(PROJECT_ROOT / "scripts").glob("*.ps1")]:
        source = path.read_text(encoding="utf-8-sig")
        for module in re.findall(r"from (grading_agent[\w.]+) import", source):
            assert importlib.util.find_spec(module) is not None, (path.name, module)


def test_workspace_root_survives_infrastructure_move():
    assert project_roots()[0] == PROJECT_ROOT


def test_opencv_flavors_are_consistent_across_extras():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    for group in ("local_recognition", "ocr", "ocr_engine"):
        assert any(
            req.startswith("opencv-contrib-python")
            for req in metadata["optional-dependencies"][group]
        )
        assert not any(
            req.startswith("opencv-python-headless")
            for req in metadata["optional-dependencies"][group]
        )


def test_source_imports_have_declared_dependencies():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    requirements = [*metadata["dependencies"]]
    for group in metadata["optional-dependencies"].values():
        requirements.extend(group)
    declared = {canonicalize_name(Requirement(text).name) for text in requirements}
    import_to_distribution = {
        "PIL": "Pillow",
        "docx": "python-docx",
        "cv2": "opencv-contrib-python",
        "paddle": "paddlepaddle",
    }
    for path in (PROJECT_ROOT / "src/grading_agent").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module.split(".")[0]]
            else:
                continue
            for module in modules:
                if module in sys.stdlib_module_names or module == "grading_agent":
                    continue
                distribution = canonicalize_name(import_to_distribution.get(module, module))
                assert distribution in declared, (path.name, module)
