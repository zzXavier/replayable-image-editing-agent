import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from src.agent import execute, replay
from src.comfy import workflow


class TestBackend:
    checkpoint = "test-only"

    def render(self, source, mask, prompt, seed, path):
        return Image.composite(
            source, Image.new("RGB", source.size, "red"), mask
        ), workflow(self.checkpoint, "fixture.png", prompt, seed)


class TestTrace(unittest.TestCase):
    def test_feedback_replay_and_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            im = Image.new("RGB", (16, 16), "white")
            m = Image.new("L", im.size)
            ImageDraw.Draw(m).rectangle((2, 2, 8, 9), fill=255)
            log = execute(
                TestBackend(), im, m, "red", "feedback", 13, out, fail_first=True
            )
            self.assertEqual(log["status"], "accepted")
            self.assertEqual(sum(x["tool"] == "render" for x in log["steps"]), 2)
            self.assertEqual(replay(out / "trace.json"), log)
            (out / "candidate-1.png").write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                replay(out / "trace.json")

    def test_invalid_mask_never_calls_tool(self):
        with tempfile.TemporaryDirectory() as d:
            im = Image.new("RGB", (8, 8))
            log = execute(
                TestBackend(), im, Image.new("L", im.size), "red", "fixed", 5, Path(d)
            )
            self.assertEqual(log["status"], "invalid_mask")
            self.assertEqual(len(log["steps"]), 1)

    def test_diffusers_loader(self):
        self.assertEqual(
            workflow("diffusers:checked-sha", "a.png", "red", 1)["1"]["inputs"],
            {"model_path": "checked-sha"},
        )
