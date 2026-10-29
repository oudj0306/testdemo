"""Exercise background launching in Windows PowerShell without PowerShell 7 on PATH."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher integration")
def test_windows_powershell_child_with_spaces(tmp_path):
    shell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    helper = Path(__file__).resolve().parents[1] / "scripts/local-vision-shell.ps1"
    folder = tmp_path / "path with spaces"
    folder.mkdir()
    child = folder / "child script.ps1"
    child.write_text('Write-Output "child-ok"', encoding="ascii")
    probe = folder / "probe.ps1"
    output = folder / "result.txt"
    # Inherit the test host's permitted policy; do not modify persistent policy.
    policy = subprocess.check_output(
        [
            str(shell),
            "-NoProfile",
            "-Command",
            "Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Security/Microsoft.PowerShell.Security.psd1'); Get-ExecutionPolicy",
        ],
        text=True,
    ).strip()
    if policy == "Restricted":
        pytest.skip("The test session does not permit PowerShell scripts")

    def quote(path):
        return "'" + str(path).replace("'", "''") + "'"

    probe.write_text(
        "$ErrorActionPreference='Stop'\n"
        f". {quote(helper)}\n"
        "$env:PATH=Join-Path $env:SystemRoot 'System32'\n"
        "$shell=Get-LocalVisionShell\n"
        "if($shell -ne (Join-Path $PSHOME 'powershell.exe')){throw 'Wrong host'}\n"
        f"$p=Start-Process -FilePath $shell -ArgumentList (Get-LocalVisionShellArguments {quote(child)}) "
        f"-WindowStyle Hidden -RedirectStandardOutput {quote(output)} -PassThru -Wait\n"
        "if($p.ExitCode -ne 0){throw 'Child failed'}\n",
        encoding="utf-8-sig",
    )
    subprocess.run(
        [str(shell), "-NoProfile", "-ExecutionPolicy", policy, "-File", str(probe)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert output.read_text().strip() == "child-ok"
