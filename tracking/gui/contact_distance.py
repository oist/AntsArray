"""Frame-local skeleton distances, independent of saved interaction detections."""

import json
from pathlib import Path

import numpy as np
from tracking.colony.skeleton_contacts import PairDistance, skeleton_pair_distances, validate_distance


class FinishedTrackClip:
    """Load only finished pose coordinates from a review cache, not its contacts."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.start = int(self.manifest["start_frame"])
        self.stop = int(self.manifest["stop_frame"])
        self.target = int(self.manifest["target"])
        self.fps = float(self.manifest["fps"])
        with np.load(self.root / "geometry.npz") as data:
            self.ids = data["ids"]
            self.xy = data["xy"]
            self.anchors = data["anchors"]
