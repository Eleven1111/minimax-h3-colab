"""One-shot IndexTTS2 subprocess. Run using its isolated environment Python."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, model = args.repo_dir.resolve(), args.model_dir.resolve()
    reference, output = args.reference.resolve(), args.output.resolve()
    text = args.text_file.read_text(encoding="utf-8").strip()
    if not text or len(text) > 2000:
        parser.error("Speech text must contain 1–2000 characters.")
    if not reference.is_file() or not (model / "config.yaml").is_file():
        parser.error("Voice reference or IndexTTS2 model config is missing.")
    if not (repo / "indextts" / "infer_v2.py").is_file():
        parser.error("The selected checkout does not contain IndexTTS2 infer_v2.py.")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)  # Upstream auxiliary caches and normalization resources use this directory.
    sys.path.insert(0, str(repo))
    import torch
    from indextts.infer_v2 import IndexTTS2

    if not torch.cuda.is_available():
        raise RuntimeError("The IndexTTS2 environment cannot access a CUDA GPU.")
    tts = IndexTTS2(cfg_path=str(model / "config.yaml"), model_dir=str(model),
                    device="cuda:0", use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
    tts.infer(spk_audio_prompt=str(reference), text=text, output_path=str(output), verbose=False)
    if not output.is_file() or output.stat().st_size <= 44:
        raise RuntimeError("IndexTTS2 did not produce a nonempty audio file.")


if __name__ == "__main__":
    main()
