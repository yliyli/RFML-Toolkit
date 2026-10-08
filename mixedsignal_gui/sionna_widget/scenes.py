"""Registry of Sionna RT scenes bundled with the package.

Each entry carries the scene XML path plus defaults that make the scene
usable on first load: a TX/RX pair placed on real geometry and a fallback
carrier frequency.  The registry is UI-free so both
``mixedsignal_gui.tabs.channel_tab`` and the standalone
``sionna_widget.controls.SimpleControlPanel`` can share it.
"""

from pathlib import Path
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

_ROOT = Path(__file__).parent

#: Scenes shipped with the package.  ``tx_position``/``rx_position`` are in
#: Sionna's Z-up world frame, chosen to sit on the ground/buildings of each
#: scene so ``PathSolver`` returns paths without any manual placement.
BUNDLED_SCENES: List[Dict[str, Any]] = [
    {
        "name": "HCRO — Hat Creek Radio Observatory",
        "path": _ROOT / "hcro" / "hcro.xml",
        # Mast above the observatory buildings (roofs top out near z = 24).
        "tx_position": [1040.0, -606.0, 25.0],
        # 600 m due east, ~1.8 m above local ground (terrain z ~ 0.2 there).
        "rx_position": [1640.0, -606.0, 2.0],
        "frequency_ghz": 3.5,
        "description": (
            "4.3 x 2.4 km terrain with observatory buildings. "
            "Ground datum z = 0 (1015 m ASL)."
        ),
    },
    {
        "name": "Austin Downtown",
        "path": _ROOT / "austin" / "Austin_Downtown.xml",
        "tx_position": [0.0, 0.0, 40.0],
        "rx_position": [150.0, 60.0, 1.5],
        "frequency_ghz": 3.5,
        "description": "Dense urban blocks on a flat ground plane.",
    },
    {
        "name": "Austin Suburban",
        "path": _ROOT / "austin" / "Austin_suburban.xml",
        # Mast well clear of the low suburban rooftops (median 6.5 m, max 17.4 m).
        "tx_position": [-150.0, -100.0, 25.0],
        # Street level in an open cell 213 m away, with buildings all around.
        "rx_position": [12.5, 37.5, 1.5],
        "frequency_ghz": 3.5,
        "description": "Low-rise suburban blocks on a flat ground plane.",
    },
    {
        "name": "UT Twin",
        "path": _ROOT / "austin" / "UT_Twin_1.xml",
        "tx_position": [0.0, 0.0, 40.0],
        "rx_position": [200.0, 150.0, 1.5],
        "frequency_ghz": 3.5,
        "description": "Campus buildings on a flat ground plane.",
    },
]


def available_scenes() -> List[Dict[str, Any]]:
    """Only offer scenes with readable XML and all referenced file assets."""
    return [s for s in BUNDLED_SCENES if _scene_assets_present(s["path"])]


def _scene_assets_present(path: Path) -> bool:
    try:
        root = ET.parse(path).getroot()
        return all(
            (path.parent / element.attrib["value"]).is_file()
            for element in root.iter()
            if element.attrib.get("name") == "filename"
        )
    except (OSError, ET.ParseError, KeyError):
        return False
