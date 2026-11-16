"""Opt-in Linux/Colab installer; leaves ComfyUI's Python and PyTorch untouched.

Run with the notebook Python. Official IndexTTS2 code and its uv lock are pinned;
uv provisions a separate Python 3.10 environment with upstream CUDA 12.8 wheels.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


SOURCE_URL = "https://github.com/index-tts/index-tts.git"
SOURCE_REVISION = "830f6f8f94a51fea23ab1d639027a86200075a4e"  # official v2.0.0
MODEL_REVISION = "740dcaff396282ffb241903d150ac011cd4b1ede"


def install(root):
    if sys.platform != "linux":
        raise RuntimeError("This installer targets Linux/Colab with an NVIDIA GPU.")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A failed reinstall may leave an incomplete runtime; never reuse an old ready marker.
    (root / "toybox_voice_config.json").unlink(missing_ok=True)
    repo = root / "source"
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    new_checkout = not (repo / ".git").is_dir()
    if new_checkout:
        subprocess.run(["git", "clone", "--no-checkout", SOURCE_URL, str(repo)], check=True, env=env)
    origin = subprocess.check_output(["git", "-C", str(repo), "remote", "get-url", "origin"], text=True).strip()
    if origin != SOURCE_URL:
        raise RuntimeError(f"Refusing to modify a checkout from another origin: {repo}")
    # Never reset an existing checkout: custom changes should fail visibly.
    if not new_checkout and subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise RuntimeError(f"IndexTTS checkout has local changes; preserve them before reinstalling: {repo}")
    subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", SOURCE_REVISION], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", SOURCE_REVISION], check=True, env=env)

    bootstrap = root / "installer-venv"
    if not (bootstrap / "bin" / "python").is_file():
        venv.EnvBuilder(with_pip=True).create(bootstrap)
    bootstrap_python = bootstrap / "bin" / "python"
    subprocess.run([str(bootstrap_python), "-m", "pip", "install", "uv>=0.8,<1"], check=True)
    uv = bootstrap / "bin" / "uv"
    # Pin the runtime Python as well as dependencies; do not inherit a notebook venv.
    uv_env = dict(env)
    uv_env.pop("VIRTUAL_ENV", None)
    uv_env["UV_PROJECT_ENVIRONMENT"] = str(root / "runtime-venv")
    subprocess.run([str(uv), "sync", "--frozen", "--python", "3.10"], cwd=repo, env=uv_env, check=True)
    python = root / "runtime-venv" / "bin" / "python"
    # Preserve the entire official IndexTTS environment while adding separation.
    # In particular, the notebook's CUDA 13 torch and this venv's CUDA 12.8 torch
    # must never be upgraded or downgraded by Demucs' broad version requirements.
    constraints = root / "indextts-constraints.txt"
    installed = subprocess.check_output([str(python), "-c", (
        "from importlib.metadata import distributions; "
        "print('\\n'.join(sorted(d.metadata['Name'] + '==' + d.version for d in distributions())))"
    )], cwd=repo, env=env, text=True)
    constraints.write_text(installed, encoding="utf-8")
    subprocess.run([str(uv), "pip", "install", "--python", str(python),
                    "--constraint", str(constraints), "demucs==4.0.1"], cwd=repo, env=uv_env, check=True)
    # Demucs' official checkpoint contains a serialized model class. PyTorch 2.8
    # defaults to weights_only=True; scope the compatibility override to this
    # trusted official-model subprocess, never to ComfyUI or the notebook.
    separation_env = dict(env, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    subprocess.run([str(python), "-c", (
        "from demucs.pretrained import get_model; get_model('htdemucs'); "
        "print('Demucs htdemucs checkpoint ready')"
    )], cwd=repo, env=separation_env, check=True)
    model = root / "checkpoints"
    download = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('IndexTeam/IndexTTS-2', revision={MODEL_REVISION!r}, local_dir={str(model)!r})"
    )
    subprocess.run([str(python), "-c", download], cwd=repo, env=env, check=True)
    subprocess.run([str(python), "-c", (
        "import torch; from indextts.infer_v2 import IndexTTS2; "
        "assert torch.cuda.is_available(), 'IndexTTS2 requires an available CUDA GPU'; "
        "print('IndexTTS2 runtime:', torch.__version__, torch.version.cuda)"
    )], cwd=repo, env=env, check=True)
    config = {"enabled": True, "python": str(python), "repo_dir": str(repo), "model_dir": str(model)}
    (root / "toybox_voice_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("IndexTTS2 + Demucs installed in an isolated environment; htdemucs is cached.")
    print("The separate Python/PyTorch environment and voice models use several extra GB of disk.")
    print("IndexTTS2 auxiliary models still download on first synthesis; CPU separation adds processing time.")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/indextts2"))
    args = parser.parse_args()
    install(args.root)


if __name__ == "__main__":
    main()
