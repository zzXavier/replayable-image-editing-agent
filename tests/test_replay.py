import tempfile
import json
import unittest
from copy import deepcopy
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
            self.assertEqual(replay(Path(d) / "trace.json"), log)

    def test_replay_rejects_forged_final_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = Image.new("RGB", (16, 16), "white")
            mask = Image.new("L", source.size)
            ImageDraw.Draw(mask).rectangle((3, 3, 9, 9), fill=255)
            trace = execute(TestBackend(), source, mask, "red", "direct", 7, out)
            trace["status"] = "rejected"
            path = out / "trace.json"
            path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "Final decision"):
                replay(path)

    def test_diffusers_loader(self):
        self.assertEqual(
            workflow("diffusers:checked-sha", "a.png", "red", 1)["1"]["inputs"],
            {"model_path": "checked-sha"},
        )

    def test_replay_binds_graph_to_request(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = Image.new("RGB", (16, 16), "white")
            mask = Image.new("L", source.size)
            ImageDraw.Draw(mask).rectangle((3, 3, 9, 9), fill=255)
            trace = execute(TestBackend(), source, mask, "red", "direct", 13, out)
            for field, value in (("prompt", "blue"), ("checkpoint", "other-model")):
                with self.subTest(field=field):
                    altered = deepcopy(trace)
                    altered[field] = value
                    path = out / "trace.json"
                    path.write_text(json.dumps(altered))
                    with self.assertRaises(ValueError):
                        replay(path)
            trace["steps"][1]["graph"]["6"]["inputs"]["steps"] = 1
            (out / "trace.json").write_text(json.dumps(trace))
            with self.assertRaises(ValueError):
                replay(out / "trace.json")

    def test_fault_trace_requires_validation_and_failed_call(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = Image.new("RGB", (16, 16), "white")
            mask = Image.new("L", source.size)
            ImageDraw.Draw(mask).rectangle((3, 3, 9, 9), fill=255)
            trace = execute(TestBackend(), source, mask, "red", "direct", 13, out, fail_first=True)
            self.assertEqual(replay(out / "trace.json"), trace)
            validate, failed = trace["steps"]
            no_error = {k: v for k, v in failed.items() if k != "error"}
            for steps in ([], [failed], [validate], [validate, no_error], [failed, validate]):
                with self.subTest(steps=steps):
                    altered = dict(trace, steps=steps)
                    (out / "trace.json").write_text(json.dumps(altered))
                    with self.assertRaises(ValueError):
                        replay(out / "trace.json")

    def test_comfy_upload_alias_and_alpha_match_the_recorded_input(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = Image.new("RGB", (16, 16), "white")
            mask = Image.new("L", source.size)
            ImageDraw.Draw(mask).rectangle((3, 3, 9, 9), fill=255)
            trace = execute(TestBackend(), source, mask, "red", "direct", 13, out)
            trace["server_source_revision"] = "local-test"
            upload = out / "candidate-0-input.png"
            packed = source.convert("RGBA")
            packed.putalpha(mask)
            packed.save(upload)
            graph_image = trace["steps"][1]["graph"]["2"]["inputs"]
            graph_image["image"] = "candidate-0-input (2).png"
            path = out / "trace.json"
            path.write_text(json.dumps(trace))
            self.assertEqual(replay(path), trace)
            graph_image["image"] = "other-input.png"
            path.write_text(json.dumps(trace))
            with self.assertRaises(ValueError):
                replay(path)
            graph_image["image"] = "candidate-0-input.png"
            path.write_text(json.dumps(trace))
            packed.putalpha(255)
            packed.save(upload)
            with self.assertRaises(ValueError):
                replay(path)

    def test_replay_requires_correct_evaluation_order_and_fixed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = Image.new("RGB", (16, 16), "white")
            mask = Image.new("L", source.size)
            ImageDraw.Draw(mask).rectangle((3, 3, 9, 9), fill=255)
            trace = execute(TestBackend(), source, mask, "red", "fixed", 5, out, fail_first=True)
            self.assertEqual(replay(out / "trace.json"), trace)
            validate, failed, rendered, evaluated = trace["steps"]
            for steps in ([validate, failed, evaluated, rendered], [*trace["steps"], evaluated]):
                (out / "trace.json").write_text(json.dumps(dict(trace, steps=steps)))
                with self.assertRaises(ValueError):
                    replay(out / "trace.json")
            shortened = {"status": "tool_error", "steps": [validate, failed]}
            altered = {k: v for k, v in trace.items() if k not in ("scores", "chosen_image")} | shortened
            (out / "trace.json").write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                replay(out / "trace.json")
