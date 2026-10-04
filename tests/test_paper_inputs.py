"""Regression checks for recorded I/Q and seeded experiment inputs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QSettings

from mixedsignal_gui.backend.dataset_manager import DatasetManager
from mixedsignal_gui.backend.sigmf_recordings import read_sigmf_regions
from mixedsignal_gui.backend.bulk_augmentation import BulkAugmentationThread
from mixedsignal_gui.backend.channel_bank import MeasuredChannel
from mixedsignal_gui.backend.parameter_range import ParameterRange
from mixedsignal_gui.backend.waveform_pipeline import WaveformPipeline


class PaperInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = np.arange(128, dtype=np.float32) + 1j * np.arange(128, dtype=np.float32)[::-1]
        self.meta = {
            "global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6},
            "captures": [{"core:sample_start": 0, "core:frequency": 900e6},
                         {"core:sample_start": 64, "core:frequency": 901e6}],
            "annotations": [{"core:sample_start": 8, "core:sample_count": 16, "core:label": "QAM"},
                            {"core:sample_start": 70, "core:sample_count": 24, "core:label": "PSK"}],
        }
        self.meta_path = self.root / "recording.sigmf-meta"
        self.data_path = self.root / "recording.sigmf-data"
        self.samples.astype("<c8").tofile(self.data_path)
        self.write_metadata()

    def write_metadata(self):
        self.meta_path.write_text(json.dumps(self.meta))

    def test_sigmf_registry_preserves_regions_and_metadata(self):
        manager = DatasetManager(str(self.root / "datasets"))
        entries = manager.import_sigmf(str(self.data_path))
        self.assertEqual(len(entries), 2)
        for entry, (start, count, label, frequency) in zip(entries, [(8, 16, "QAM", 900e6), (70, 24, "PSK", 901e6)]):
            np.testing.assert_array_equal(manager.load_signal(entry), self.samples[start:start + count])
            self.assertEqual(entry["modulation"], label)
            self.assertEqual(entry["fc"], frequency)
            self.assertEqual(entry["samples"], count)
            self.assertEqual(entry["sample_start"], start)
            self.assertEqual(entry["source"], "sigmf_recording")
            self.assertEqual(entry["sigmf"]["global"], self.meta["global"])
        self.assertEqual(len(manager.scan()), 2)

    def test_sigmf_capture_regions_and_boundary_crossing(self):
        self.meta["annotations"] = []
        self.write_metadata()
        regions = read_sigmf_regions(self.meta_path)
        self.assertEqual([len(x) for x, _ in regions], [64, 64])
        self.meta["annotations"] = [{"core:sample_start": 60, "core:sample_count": 10, "core:label": "QAM"}]
        self.write_metadata()
        regions = read_sigmf_regions(self.meta_path)
        np.testing.assert_array_equal(np.concatenate([x for x, _ in regions]), self.samples[60:70])
        self.assertEqual([m["fc"] for _, m in regions], [900e6, 901e6])
        self.meta["annotations"] = [{"core:sample_start": 60}]
        self.write_metadata()
        self.assertEqual(len(read_sigmf_regions(self.meta_path)[0][0]), 4)

    def test_sigmf_integer_iq_and_big_endian(self):
        iq = np.arange(256, dtype=np.int16).reshape(128, 2)
        self.meta["global"]["core:datatype"] = "ci16_be"
        iq.astype(">i2").tofile(self.data_path)
        self.write_metadata()
        signal, _ = read_sigmf_regions(self.meta_path)[0]
        np.testing.assert_array_equal(signal, iq[8:24, 0] + 1j * iq[8:24, 1])

    def test_sigmf_rejects_invalid_inputs_without_partial_import(self):
        manager = DatasetManager(str(self.root / "invalid"))
        for mutate in (lambda m: m["global"].pop("core:sample_rate"),
                       lambda m: m["global"].update({"core:datatype": "rf32_le"}),
                       lambda m: m["annotations"][1].update({"core:sample_count": 1000}),
                       lambda m: m["captures"][0].update({"core:header_bytes": 4})):
            original = copy.deepcopy(self.meta)
            mutate(self.meta)
            self.write_metadata()
            with self.assertRaises(ValueError):
                manager.import_sigmf(str(self.meta_path))
            self.assertEqual(manager.scan(), [])
            self.meta = original

    def bulk(self, folder, seed, kind="awgn"):
        source = DatasetManager(str(self.root / "source"))
        if not source.scan():
            for i in range(3):
                source.save(f"signal{i}", self.samples, {"fs": 1e6, "source": "test", "modulation": "QAM"})
        dest = DatasetManager(str(self.root / folder))
        ranges = {"snr_db": ParameterRange.uniform(5, 20),
                  "meas_snr_db": ParameterRange.uniform(5, 20)}
        worker = BulkAugmentationThread(source, dest, kind, ranges,
            {"random_channel": True, "meas_awgn_enabled": True}, seed=seed,
            measured_channels=[MeasuredChannel("direct", np.array([1+0j]), fs=1e6),
                               MeasuredChannel("echo", np.array([1+0j, .5j]), fs=1e6)])
        finished = []
        worker.finished.connect(lambda *args: finished.append(args))
        worker.run()
        self.assertEqual(finished[0][:2], (3, 0))
        return {e["original_name"]: (dest.load_signal(e), e) for e in dest.scan()}

    def test_bulk_noise_and_measured_channels_repeat_from_recorded_seed(self):
        for kind in ("awgn", "measured"):
            first = self.bulk(kind + "a", 42, kind)
            repeat = self.bulk(kind + "b", 42, kind)
            changed = self.bulk(kind + "c", 43, kind)
            for name, (samples, metadata) in first.items():
                np.testing.assert_array_equal(samples, repeat[name][0])
                self.assertEqual(metadata["run_seed"], 42)
                self.assertEqual(metadata["entry_seed"], 42 + metadata["entry_index"])
                self.assertEqual(metadata["augmentation_seed"], repeat[name][1]["augmentation_seed"])
                self.assertEqual(metadata["augmentation_config"], repeat[name][1]["augmentation_config"])
                self.assertFalse(np.array_equal(samples, changed[name][0]))

    def test_cpu_training_seed_repeats_weights_and_learning_curve(self):
        import torch
        from mixedsignal_gui.backend.trainer import TrainerThread
        files = []
        for i in range(8):
            path = self.root / f"train{i}.npy"
            np.save(path, self.samples * np.exp(1j * i))
            files.append((str(path), i % 2))
        def train(folder, seed):
            worker = TrainerThread(files, ["QAM", "PSK"], model_name="TinyConv",
                epochs=1, batch_size=4, save_dir=str(self.root / folder), seed=seed)
            worker.TARGET_LENGTH = 32
            paths, curve = [], []
            worker.finished.connect(paths.append)
            worker.progress.connect(lambda *args: curve.append(args))
            worker.run()
            self.assertTrue(paths[0], worker.error)
            metadata = json.loads(Path(paths[0]).with_suffix(".json").read_text())
            self.assertEqual(metadata["run_seed"], seed)
            return torch.load(paths[0], weights_only=True), curve
        first, curve = train("train_a", 42)
        repeat, repeat_curve = train("train_b", 42)
        changed, _ = train("train_c", 43)
        self.assertEqual(curve, repeat_curve)
        self.assertTrue(all(torch.equal(first[k], repeat[k]) for k in first))
        self.assertTrue(any(not torch.equal(first[k], changed[k]) for k in first))

    def test_waveform_runs_repeat_but_examples_differ(self):
        args = dict(fs=8e6, Tsymb=1e-6, Nsymb=16, fc=1e6, M=4,
                    modulation="QAM", var=0, alpha=.35, span=8, pulse_shape="rrc")
        first, repeat = WaveformPipeline(None, seed=91), WaveformPipeline(None, seed=91)
        a, b = first.generate(**args), first.generate(**args)
        np.testing.assert_array_equal(a["signal"], repeat.generate(**args)["signal"])
        np.testing.assert_array_equal(b["signal"], repeat.generate(**args)["signal"])
        self.assertFalse(np.array_equal(a["signal"], b["signal"]))
        self.assertEqual(b["seed_metadata"]["generation_seed"], 92)

    def test_settings_seed_persists_and_is_read_by_new_runs(self):
        from mixedsignal_gui.backend.experiment_settings import experiment_seed
        settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
        with patch("mixedsignal_gui.backend.experiment_settings.QSettings", return_value=settings):
            self.assertEqual(experiment_seed(), 42)
            settings.setValue("experimentSeed", 123)
            settings.sync()
            self.assertEqual(experiment_seed(), 123)
            self.assertEqual(WaveformPipeline(None).run_seed, 123)


if __name__ == "__main__":
    unittest.main()
