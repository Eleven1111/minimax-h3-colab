# IndexTTS2 voice cloning verification

## Check: request validation and job lifecycle
**Evidence type:** Runtime tests with stubbed ComfyUI/model calls.
**Command or source:** `python3 -m unittest discover -s toybox/tests -v`
**Observation:** On the combined working branch, 20 tests passed, including the 5 existing director tests; the standalone voice-cloning PR has 15 passing tests and excludes the pending director feature. New checks cover explicit dialogue, reference duration/type, extension rejection, disabled setup, one-time dubbing dispatch after persistence, GPU submission gating, unloading before synthesis, deleted references and restart recovery. Tests exercise real server functions extracted without importing unavailable GPU/media dependencies.
**Result: PASS** for the tested orchestration. Actual CUDA memory release is not established by these stubs.

## Check: video and audio composition
**Evidence type:** Runtime with actual ffmpeg/ffprobe and generated test signals.
**Command or source:** `toybox/tests/test_voice_clone.py`, `SoundtrackTest`.
**Observation:** Frame checksums before/after are identical. A 2-second output retains its duration, contains the supplied background and new speech frequencies, and excludes the old test tone. Overlong speech fails instead of truncating dialogue; failed synthesis preserves the original and cleans temporary files.
**Result: PASS** for composition. The background input in this test is a fixture, not a real Demucs separation result.

## Check: upload, form and request wiring
**Evidence type:** Runtime browser and HTTP checks against a local fixture server.
**Command or source:** Toybox served by its real HTTP handler with media imports/Comfy endpoints stubbed; ego-browser exercised audio upload and the generation form.
**Observation:** Uploaded a generated 4-second WAV, enabled voice cloning, filled the script and submitted generation. The browser and stored job both contained `enabled: true`, the uploaded asset ID, the exact script and `keep_background: true`; the voice reference stayed outside the H3 tray. `GET /api/config` exposed availability; invalid script returned HTTP 400. Expanded form was visually inspected after widening it to span the options grid.
**Result: PASS** for the UI/API contract. No generated human voice or video was produced by this fixture.

## Check: installer and runner contract
**Evidence type:** Static inspection and runtime tests with subprocess/inference stubs.
**Command or source:** `toybox/tests/test_voice_setup.py`; syntax checks on Python files, every notebook code cell, `node --check toybox/static/app.js`, and `git diff --check`.
**Observation:** Installer commands target a separate Python 3.10 environment, pin official IndexTTS2 source/model revisions, constrain existing package versions when adding Demucs, and publish no new ready manifest after a failed first installation. Runner passes reference/text/output through the verified IndexTTS2 API. Syntax/whitespace checks passed.
**Result: PASS** for these contracts; fresh dependency installation has not been executed on Colab.

## Required GPU validation still pending

- Run a clean L4 Colab setup with `INSTALL_VOICE_CLONE` enabled; check extra disk consumption, first-run auxiliary downloads and dependencies.
- Upload a real authorized 3–30 second recording and photo, use a short explicit script, and generate a 5–10 second video.
- Confirm H3 unloads, IndexTTS2 runs without OOM, Demucs retains usable music/effects, and both original and dubbed versions play and enter the library correctly.
- Judge voice similarity, intelligibility, residual original speech and timing. This implementation does not include lip-sync correction or sentence-level alignment.

No actual CUDA inference, real source separation or voice-similarity evaluation was possible in the local macOS environment. The code is reviewable; the GPU path requires this follow-up before a production-ready claim.

VERDICT: PARTIAL
