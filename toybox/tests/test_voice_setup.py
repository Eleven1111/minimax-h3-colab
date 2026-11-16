"""Check setup and one-shot inference contracts without downloads or GPU imports."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load_module("voice_installer", "tools/install_indextts2.py")
runner = load_module("voice_runner", "toybox/voice_runner.py")


class VoiceSetupTests(unittest.TestCase):
    def test_installer_keeps_comfy_environment_and_pins_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def command_output(command, **kwargs):
                if command[0] == "git":
                    return installer.SOURCE_URL + "\n"
                return "torch==2.8.0+cu128\ntorchaudio==2.8.0+cu128\nnumpy==1.26.2\n"

            with patch.object(installer.sys, "platform", "linux"), \
                    patch.object(installer.venv, "EnvBuilder"), \
                    patch.object(installer.subprocess, "run") as run, \
                    patch.object(installer.subprocess, "check_output", side_effect=command_output), \
                    contextlib.redirect_stdout(io.StringIO()):
                cfg = installer.install(root)
            calls = run.call_args_list
            commands = [call.args[0] for call in calls]
            self.assertIn(["git", "-C", str(root / "source"), "checkout", "--detach",
                           "830f6f8f94a51fea23ab1d639027a86200075a4e"], commands)
            sync = next(call for call in calls if "sync" in call.args[0])
            self.assertIn("--frozen", sync.args[0])
            self.assertIn("3.10", sync.args[0])
            self.assertEqual(sync.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(root / "runtime-venv"))
            self.assertNotIn("VIRTUAL_ENV", sync.kwargs["env"])
            bootstrap = next(command for command in commands if "uv>=0.8,<1" in command)
            self.assertEqual(bootstrap[0], str(root / "installer-venv/bin/python"))
            demucs = next(command for command in commands if "demucs==4.0.1" in command)
            self.assertEqual(demucs[demucs.index("--python") + 1], str(root / "runtime-venv/bin/python"))
            self.assertIn("--constraint", demucs)
            self.assertIn("torch==2.8.0+cu128", (root / "indextts-constraints.txt").read_text())
            prefetch = next(call for call in calls if any("get_model('htdemucs')" in arg for arg in call.args[0]))
            self.assertEqual(prefetch.kwargs["env"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
            download = next(command for command in commands if any("snapshot_download" in arg for arg in command))
            self.assertIn(installer.MODEL_REVISION, download[-1])
            self.assertEqual(cfg["python"], str(root / "runtime-venv/bin/python"))
            self.assertEqual(cfg["model_dir"], str(root / "checkpoints"))
            self.assertTrue((root / "toybox_voice_config.json").is_file())

    def test_failed_install_does_not_publish_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(installer.sys, "platform", "linux"), \
                    patch.object(installer.subprocess, "run", side_effect=RuntimeError("download unavailable")):
                with self.assertRaises(RuntimeError):
                    installer.install(root)
            self.assertFalse((root / "toybox_voice_config.json").exists())

    def test_runner_calls_verified_api_in_isolated_process_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "indextts").mkdir()
            (root / "indextts/infer_v2.py").touch()
            (root / "config.yaml").touch()
            (root / "reference.wav").write_bytes(b"reference")
            (root / "text.txt").write_text("你好，这是我的声音。", encoding="utf-8")
            output = root / "cloned.wav"
            calls = []

            class FakeTTS:
                def __init__(self, **kwargs):
                    calls.append(kwargs)

                def infer(self, **kwargs):
                    calls.append(kwargs)
                    Path(kwargs["output_path"]).write_bytes(b"x" * 45)

            fake = ModuleType("indextts.infer_v2")
            fake.IndexTTS2 = FakeTTS
            argv = ["voice_runner.py", "--repo-dir", str(root), "--model-dir", str(root),
                    "--reference", str(root / "reference.wav"), "--text-file", str(root / "text.txt"),
                    "--output", str(output)]
            cwd = Path.cwd()
            try:
                with patch.dict(sys.modules, {
                    "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                    "indextts.infer_v2": fake,
                }), patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)):
                    runner.main()
            finally:
                os.chdir(cwd)
            self.assertTrue(output.is_file())
            self.assertTrue(calls[0]["use_fp16"])
            self.assertFalse(calls[0]["use_deepspeed"])
            self.assertFalse(calls[0]["use_cuda_kernel"])
            self.assertEqual(calls[1]["text"], "你好，这是我的声音。")
            self.assertEqual(calls[1]["spk_audio_prompt"], str(root / "reference.wav"))


if __name__ == "__main__":
    unittest.main()
