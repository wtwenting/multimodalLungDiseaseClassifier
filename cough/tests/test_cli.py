import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from psupr_proj.cli import main
from psupr_proj.data import analyze_cough_thresholds, filter_dataset


class TestCli(unittest.TestCase):
    def test_main(self):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch(
            "sys.argv", ["psupr-proj", "--data", "tests/fixtures/metadata.csv"]
        ):
            main()
        self.assertIn("Records: 2", output.getvalue())

    def test_filter_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            output_dir = root / "processed"
            source_dir.mkdir()
            metadata = source_dir / "metadata.csv"
            metadata.write_text(
                ",uuid,cough_detected\n0,low,0.7\n1,high,0.8\n",
                encoding="utf-8",
            )
            (source_dir / "low.wav").write_bytes(b"low")
            (source_dir / "high.wav").write_bytes(b"high")
            (source_dir / "high.json").write_text("{}", encoding="utf-8")

            result = filter_dataset(metadata, source_dir, output_dir)

            self.assertEqual(result, {"records": 1, "files": 2, "missing_files": 0})
            self.assertTrue((output_dir / "high.wav").exists())
            self.assertTrue((output_dir / "high.json").exists())
            self.assertFalse((output_dir / "low.wav").exists())
            self.assertIn("high", (output_dir / "metadata_filtered.csv").read_text())

    def test_filter_threshold_is_configurable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            source_dir.mkdir()
            metadata = source_dir / "metadata.csv"
            metadata.write_text(
                "uuid,cough_detected\nlow,0.7\nhigh,0.8\n",
                encoding="utf-8",
            )
            (source_dir / "low.wav").write_bytes(b"low")
            (source_dir / "high.wav").write_bytes(b"high")

            result = filter_dataset(
                metadata,
                source_dir,
                root / "processed",
                threshold=0.7,
            )

            self.assertEqual(result["records"], 2)

    def test_analyze_cough_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata = Path(temporary_directory) / "metadata.csv"
            metadata.write_text(
                "status,cough_detected\n"
                "healthy,0.75\n"
                "healthy,0.85\n"
                "COVID-19,0.75\n"
                "COVID-19,0.95\n"
                "symptomatic,0.99\n",
                encoding="utf-8",
            )

            results = analyze_cough_thresholds(
                metadata,
                thresholds=(0.70, 0.80),
            )

            self.assertEqual(results[0]["healthy"], 2)
            self.assertEqual(results[0]["COVID-19"], 2)
            self.assertEqual(results[1]["healthy"], 1)
            self.assertEqual(results[1]["COVID-19"], 1)
