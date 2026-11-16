import ast
import array
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import voice_clone


def tone(path, seconds, frequency=440):
    samples = array.array("h", (int(9000 * math.sin(2 * math.pi * frequency * i / 24000))
                               for i in range(int(seconds * 24000))))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(samples.tobytes())


class VoiceValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "config.yaml").touch()
        tone(self.root / "reference.wav", 4)
        self.cfg = {"voice_clone": {"enabled": True, "python": sys.executable,
                                    "model_dir": str(self.root)}}
        self.req = {"mode": "ref", "voice_clone": {"enabled": True, "reference_id": "mine", "text": "你好"}}
        self.get_asset = Mock(return_value={"kind": "audio", "file": "reference.wav"})

    def validate(self):
        return voice_clone.validate_request(self.cfg, self.req, self.get_asset, self.root)

    def test_disabled_does_not_require_tools_or_models(self):
        self.assertIsNone(voice_clone.validate_request({}, {}, self.get_asset, self.root))
        self.get_asset.assert_not_called()

    def test_clone_script_is_authoritative_before_prompt_rewriting(self):
        original = '她說：「舊台詞。」'
        scene = voice_clone.scene_prompt(original, self.req["voice_clone"])
        self.assertIn("replace any other spoken dialogue", scene)
        self.assertIn("Dialogue: 你好", scene)
        self.assertIn("official H3 output format", scene)
        self.assertEqual(voice_clone.scene_prompt(original, None), original)

    def test_valid_audio_defaults_to_preserving_background(self):
        result = self.validate()
        self.assertTrue(result["keep_background"])
        self.assertEqual(result["text"], "你好")

    def test_empty_text_wrong_reference_short_audio_and_extension_rejected(self):
        mutations = [
            ({"text": "  "}, {}, {"kind": "audio", "file": "reference.wav"}),
            ({"text": "x" * 2001}, {}, {"kind": "audio", "file": "reference.wav"}),
            ({}, {}, {"kind": "image", "file": "reference.wav"}),
            ({}, {"tray": [{"extend": True}]}, {"kind": "audio", "file": "reference.wav"}),
            ({}, {"mode": "image"}, {"kind": "audio", "file": "reference.wav"}),
        ]
        for voice, request, asset in mutations:
            with self.subTest(voice=voice, request=request):
                req = dict(self.req, **request)
                req["voice_clone"] = dict(self.req["voice_clone"], **voice)
                with self.assertRaises(ValueError):
                    voice_clone.validate_request(self.cfg, req, lambda _: asset, self.root)
        tone(self.root / "reference.wav", 1)
        with self.assertRaisesRegex(ValueError, "3–30"):
            self.validate()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class SoundtrackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "original.mp4"
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=blue:s=64x64:r=24:d=2", "-f", "lavfi", "-i",
                        "sine=frequency=800:duration=2", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(self.video)], check=True, capture_output=True)
        self.speech = self.root / "speech.wav"
        self.bg = self.root / "background.wav"
        tone(self.speech, 1, 440)
        tone(self.bg, 2, 220)

    def audio_energy(self, path, frequency):
        data = subprocess.check_output([shutil.which("ffmpeg"), "-v", "error", "-i", str(path),
                                        "-vn", "-ac", "1", "-ar", "24000", "-f", "s16le", "-"])
        values = array.array("h", data)[:24000]
        return abs(sum(value * complex(math.cos(2 * math.pi * frequency * i / 24000),
                                        math.sin(2 * math.pi * frequency * i / 24000))
                       for i, value in enumerate(values)))

    def test_real_mux_preserves_frames_background_and_new_voice(self):
        output = self.root / "cloned.mp4"
        voice_clone.replace_soundtrack(self.video, self.speech, output, self.bg)
        self.assertAlmostEqual(voice_clone.media_duration(output), 2, delta=0.06)
        old = self.audio_energy(output, 800)
        self.assertGreater(self.audio_energy(output, 440), old * 20)
        self.assertGreater(self.audio_energy(output, 220), old * 20)
        def frames(path):
            return subprocess.check_output([shutil.which("ffmpeg"), "-v", "error", "-i", str(path),
                                             "-map", "0:v", "-f", "framemd5", "-"])
        self.assertEqual(frames(self.video), frames(output))

    def test_overlong_speech_fails_without_output(self):
        tone(self.speech, 3)
        output = self.root / "cloned.mp4"
        with self.assertRaisesRegex(ValueError, "縮短台詞"):
            voice_clone.replace_soundtrack(self.video, self.speech, output, self.bg)
        self.assertFalse(output.exists())

    def test_entire_soundtrack_replacement_is_optional(self):
        output = self.root / "cloned.mp4"
        voice_clone.replace_soundtrack(self.video, self.speech, output)
        self.assertGreater(self.audio_energy(output, 440), self.audio_energy(output, 220) * 20)

    def test_failed_tts_leaves_original_and_cleans_temporary_files(self):
        runner = self.root / "fail.py"
        runner.write_text("raise SystemExit(1)")
        before = self.video.read_bytes()
        destination = self.root / "result.mp4"
        with self.assertRaisesRegex(RuntimeError, "IndexTTS2"):
            voice_clone.synthesize_and_replace(
                {"python": sys.executable, "repo_dir": str(self.root), "model_dir": str(self.root)},
                self.speech, "台词", self.video, destination, runner,
            )
        self.assertEqual(self.video.read_bytes(), before)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob("h3voice-*")), [])


def server_functions(*names):
    tree = ast.parse((ROOT / "server.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"Path": Path, "threading": threading, "lock": threading.Lock(), "time": time,
             "tempfile": tempfile, "voice_clone": voice_clone}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "server.py"), "exec"), scope)
    return scope


class VoiceJobLifecycleTest(unittest.TestCase):
    def test_restart_marks_interrupted_dubbing_without_touching_completed_jobs(self):
        scope = server_functions("recover_voice_jobs")
        jobs = [{"id": "one", "status": "dubbing", "video": {"filename": "original.mp4"}},
                {"id": "two", "status": "done", "cloned_video": "finished.mp4"}]
        scope.update(JOBS_PATH="jobs", load_json=lambda *_: jobs, save_json=Mock())
        scope["recover_voice_jobs"]()
        self.assertEqual(jobs[0]["status"], "voice_error")
        self.assertEqual(jobs[1]["status"], "done")
        self.assertEqual(jobs[0]["video"]["filename"], "original.mp4")

    def test_worker_unloads_comfy_before_synthesis_and_releases_gpu_gate(self):
        scope = server_functions("voice_worker")
        job = {"id": "p1", "status": "dubbing", "video": {"filename": "original.mp4"},
               "voice_clone": {"reference_id": "mine", "text": "你好", "keep_background": True}}
        gate = threading.Lock()
        events = []
        def comfy(cfg, path, payload=None):
            self.assertTrue(gate.locked())
            events.append(path)
            if path == "/system_stats":
                return {"devices": [{"type": "cuda", "torch_vram_total": 0, "vram_free": 24 * 2**30}]}
            return {"queue_running": [], "queue_pending": []}
        def synth(*args, **kwargs):
            self.assertTrue(gate.locked())
            self.assertEqual(events, ["/queue", "/free", "/system_stats"])
            self.assertTrue(kwargs["keep_background"])
        with tempfile.TemporaryDirectory() as directory:
            scope.update(OUTPUTS=Path(directory), LIBRARY=Path(directory), ROOT=ROOT,
                         JOBS_PATH="jobs", GPU_LOCK=gate, comfy=comfy,
                         load_json=lambda *_: [job], save_json=Mock(),
                         asset=lambda _: {"file": "reference.wav"},
                         comfy_download=lambda cfg, video, path: path.write_bytes(b"original"))
            with patch.object(voice_clone, "synthesize_and_replace", side_effect=synth):
                scope["voice_worker"]({"voice_clone": {}}, job)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["cloned_video"], "p1_voice.mp4")
        self.assertFalse(gate.locked())

    def test_poll_persists_dubbing_before_launch_and_does_not_launch_twice(self):
        scope = server_functions("_refresh_jobs")
        job = {"id": "p1", "status": "queued", "created": 0, "voice_clone": {"enabled": True}}
        store = [job]
        def save(path, jobs):
            store[:] = json.loads(json.dumps(jobs))
        scope.update(load_json=lambda path, default: json.loads(json.dumps(store)) if path == "jobs" else {},
                     save_json=save, JOBS_PATH="jobs", STATS_PATH="stats", DEFAULT_STATS={},
                     comfy=lambda *_: {}, MONITOR=Mock(snapshot=lambda _: None),
                     history_entry=lambda *_: {"status": {}}, saved_video=lambda _: {"filename": "v.mp4"},
                     voice_worker=Mock())
        launch = Mock()
        launch.start.side_effect = lambda: self.assertEqual(store[0]["status"], "dubbing")
        with patch.object(threading, "Thread", return_value=launch) as worker:
            scope["_refresh_jobs"]({})
            scope["_refresh_jobs"]({})
        self.assertEqual(worker.call_count, 1)

    def test_missing_reference_marks_voice_error_and_retains_original(self):
        scope = server_functions("voice_worker")
        job = {"id": "p1", "status": "dubbing", "video": {"filename": "original.mp4"},
               "voice_clone": {"reference_id": "gone"}}
        store = [job]
        with tempfile.TemporaryDirectory() as directory:
            scope.update(OUTPUTS=Path(directory), JOBS_PATH="jobs", load_json=lambda *_: store,
                         save_json=lambda *args: None, asset=Mock(side_effect=ValueError("素材不存在")))
            scope["voice_worker"]({}, job)
        self.assertEqual(store[0]["status"], "voice_error")
        self.assertEqual(store[0]["video"]["filename"], "original.mp4")
        self.assertIn("素材不存在", store[0]["voice_error"])


if __name__ == "__main__":
    unittest.main()
