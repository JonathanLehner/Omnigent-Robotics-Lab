"""Regression checks for batch-start run provenance."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from lab import runner


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_fingerprint_includes_untracked_sources_and_native_extension():
    with tempfile.TemporaryDirectory() as temporary:
        tmp_path = Path(temporary)
        root = tmp_path / "repo"
        root.mkdir()
        (root / "lab").mkdir()
        (root / "methods").mkdir()
        (root / "prompts").mkdir()
        (root / "lab" / "tracked.py").write_text("TRACKED = 1\n")
        _git(root, "init")
        _git(root, "add", "lab/tracked.py")
        _git(
            root,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "initial",
        )
        untracked = root / "methods" / "new_method.yaml"
        untracked.write_text("name: new_method\n")

        sumo_env = tmp_path / "sumo-env"
        extension = (
            sumo_env
            / "lib/python3.13/site-packages/mujoco_extensions/policy_rollout"
            / "policy_rollout_pybind.cpython-313-darwin.so"
        )
        extension.parent.mkdir(parents=True)
        extension.write_bytes(b"native-v1")
        with (
            patch.object(runner, "ROOT", root),
            patch.object(runner, "SUMO_ENV", sumo_env),
        ):
            first = runner.capture_run_fingerprint()
            assert "methods/new_method.yaml" in first["workspace_files"]
            assert first["sumo_native_extensions"] == [
                {
                    "path": str(extension.relative_to(sumo_env)),
                    "mtime_ns": extension.stat().st_mtime_ns,
                    "sha256": runner._sha256_file(extension),
                }
            ]
            assert runner.capture_run_fingerprint()["id"] == first["id"]

            untracked.write_text("name: changed\n")
            second = runner.capture_run_fingerprint()
            assert first["workspace_sha256"] != second["workspace_sha256"]
            assert first["id"] != second["id"]


if __name__ == "__main__":
    test_fingerprint_includes_untracked_sources_and_native_extension()
    print("runner provenance test passed")
