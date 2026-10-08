"""Scene-menu asset checks without importing the native RT runtime."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mixedsignal_gui.sionna_widget.scenes import BUNDLED_SCENES, available_scenes


class SceneAssetsTests(unittest.TestCase):
    def test_shipped_scenes_have_all_assets(self):
        self.assertEqual(available_scenes(), BUNDLED_SCENES)

    def test_missing_or_broken_assets_are_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "scene.xml"
            scene = {"name": "test", "path": xml}
            with patch("mixedsignal_gui.sionna_widget.scenes.BUNDLED_SCENES", [scene]):
                self.assertEqual(available_scenes(), [])
                xml.write_text('<scene><string name="filename" value="mesh.ply"/></scene>')
                self.assertEqual(available_scenes(), [])
                (root / "mesh.ply").touch()
                self.assertEqual(available_scenes(), [scene])
                xml.write_text('<scene>')
                self.assertEqual(available_scenes(), [])
                xml.write_text('<scene><string name="filename"/></scene>')
                self.assertEqual(available_scenes(), [])
