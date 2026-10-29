"""Read-only dependency audit; never starts OCR, installs packages or calls APIs."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grading_agent.infrastructure.dependency_checks import check_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("basic", "local", "cpu", "gpu"), default="local")
    parser.add_argument("--include-dev", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    result = check_environment(project, args.profile, include_dev=args.include_dev)
    if args.json:
        print(json.dumps(result, ensure_ascii=True))
    else:
        print(f"Profile: {args.profile}; {'OK' if result['ok'] else 'NEEDS REPAIR'}")
        for row in result["packages"]:
            print(
                f"  {row['status']}: {row['requirement']} (installed: {row['installed'] or 'missing'})"
            )
        for conflict in result["conflicts"]:
            print("  Shared-module conflict: " + ", ".join(conflict))
        print(
            "Ollama/Word/LibreOffice are external tools; model weights are not Python dependencies."
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
