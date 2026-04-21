"""DICT_4X4_1000 bit-pattern utilities for NN-based ArUco decoding.

Extracts all 1000 marker bit patterns from OpenCV's dictionary,
precomputes 4 rotations, and provides fast Hamming-distance matching.

Usage:
    from aruco_detection.nn_detection.dict_4x4_1000 import ArucoDictionary

    d = ArucoDictionary()
    marker_id, hamming_dist, rotation = d.match_bits(predicted_bits_4x4)
"""

from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoDictionary:
    """DICT_4X4_1000 lookup table with 4-rotation Hamming matching."""

    N_MARKERS = 1000
    GRID_SIZE = 4  # 4x4 data bits
    N_BITS = 16

    def __init__(self):
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        # (N_MARKERS, 4, 4) binary arrays
        self._patterns = self._extract_all_patterns()
        # (N_MARKERS * 4, 16) flattened — all 4 rotations for fast vectorised match
        self._lookup, self._lookup_ids, self._lookup_rots = self._build_lookup()

    def _extract_all_patterns(self) -> np.ndarray:
        """Extract (1000, 4, 4) binary patterns from OpenCV dictionary.

        Uses generateImageMarker at minimal size (6x6 = 1px border + 4x4 data)
        to reliably extract the bit pattern regardless of bytesList packing.
        """
        patterns = np.zeros((self.N_MARKERS, self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        # Marker image at size=6 gives 1px border + 4x4 data region
        render_size = self.GRID_SIZE + 2
        for mid in range(self.N_MARKERS):
            img = aruco.generateImageMarker(self._aruco_dict, mid, render_size)
            data = img[1 : render_size - 1, 1 : render_size - 1]
            # Black cells (0) = bit 1, white cells (255) = bit 0
            patterns[mid] = (data < 128).astype(np.uint8)
        return patterns

    def _build_lookup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build flattened lookup table with all 4 rotations per marker.

        Returns:
            lookup: (4000, 16) uint8 flattened bit patterns
            ids:    (4000,) int32 marker IDs
            rots:   (4000,) int32 rotation index (0, 1, 2, 3)
        """
        n = self.N_MARKERS * 4
        lookup = np.zeros((n, self.N_BITS), dtype=np.uint8)
        ids = np.zeros(n, dtype=np.int32)
        rots = np.zeros(n, dtype=np.int32)

        for mid in range(self.N_MARKERS):
            pat = self._patterns[mid]
            for r in range(4):
                idx = mid * 4 + r
                rotated = np.rot90(pat, k=r)
                lookup[idx] = rotated.flatten()
                ids[idx] = mid
                rots[idx] = r

        return lookup, ids, rots

    def match_bits(
        self, bits: np.ndarray, max_distance: int = 2
    ) -> tuple[int, int, int]:
        """Match a predicted (4, 4) or (16,) bit pattern against the dictionary.

        Parameters
        ----------
        bits : np.ndarray
            Predicted binary pattern, shape (4, 4) or (16,).
        max_distance : int
            Maximum Hamming distance to accept a match.

        Returns
        -------
        marker_id : int
            Matched marker ID, or -1 if no match within max_distance.
        hamming_dist : int
            Hamming distance to the best match.
        rotation : int
            Number of 90-degree rotations of the match (0-3).
        """
        flat = bits.flatten().astype(np.uint8)
        dists = np.sum(self._lookup != flat, axis=1)
        best_idx = int(np.argmin(dists))
        best_dist = int(dists[best_idx])

        if best_dist > max_distance:
            return -1, best_dist, 0

        return int(self._lookup_ids[best_idx]), best_dist, int(self._lookup_rots[best_idx])

    def match_bits_batch(
        self, bits_batch: np.ndarray, max_distance: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch version of match_bits.

        Parameters
        ----------
        bits_batch : np.ndarray
            Shape (N, 4, 4) or (N, 16) predicted binary patterns.

        Returns
        -------
        ids : (N,) int32 — marker IDs (-1 if no match)
        dists : (N,) int32 — Hamming distances
        rots : (N,) int32 — rotation indices
        """
        N = bits_batch.shape[0]
        flat = bits_batch.reshape(N, self.N_BITS).astype(np.uint8)

        # (N, 4000) Hamming distances via broadcasting
        dists_all = np.sum(flat[:, None, :] != self._lookup[None, :, :], axis=2)
        best_indices = np.argmin(dists_all, axis=1)
        best_dists = dists_all[np.arange(N), best_indices]

        ids = self._lookup_ids[best_indices].copy()
        rots = self._lookup_rots[best_indices].copy()
        ids[best_dists > max_distance] = -1

        return ids, best_dists, rots

    def match_bits_whitelist(
        self,
        bits: np.ndarray,
        whitelist: set[int],
        max_distance: int = 2,
        min_margin: int = 2,
    ) -> tuple[int, int, int, int]:
        """Match bits against a restricted set of valid marker IDs.

        Parameters
        ----------
        bits : np.ndarray
            Predicted binary pattern, shape (4, 4) or (16,).
        whitelist : set[int]
            Only these marker IDs are considered valid matches.
        max_distance : int
            Maximum Hamming distance to accept a match.
        min_margin : int
            Minimum gap between best and runner-up Hamming distance.
            If the margin is smaller, the match is rejected (ambiguous).

        Returns
        -------
        marker_id : int
            Matched ID, or -1 if no confident match.
        hamming_dist : int
            Hamming distance to the best match.
        rotation : int
            Number of 90-degree rotations (0-3).
        margin : int
            Difference between runner-up and best Hamming distance.
        """
        flat = bits.flatten().astype(np.uint8)
        dists = np.sum(self._lookup != flat, axis=1)  # (4000,)

        # Mask entries not in the whitelist
        wl_mask = np.array(
            [self._lookup_ids[i] in whitelist for i in range(len(self._lookup_ids))],
            dtype=bool,
        )
        wl_dists = np.where(wl_mask, dists, self.N_BITS + 1)

        best_idx = int(np.argmin(wl_dists))
        best_dist = int(wl_dists[best_idx])

        if best_dist > max_distance:
            return -1, best_dist, 0, 0

        # Compute margin: distance to runner-up (different ID)
        best_id = int(self._lookup_ids[best_idx])
        other_mask = wl_mask & (self._lookup_ids != best_id)
        if np.any(other_mask):
            runner_up_dist = int(np.min(np.where(other_mask, dists, self.N_BITS + 1)))
        else:
            runner_up_dist = self.N_BITS  # only one whitelisted ID

        margin = runner_up_dist - best_dist

        if margin < min_margin:
            return -1, best_dist, 0, margin

        return (
            best_id,
            best_dist,
            int(self._lookup_rots[best_idx]),
            margin,
        )

    def get_pattern(self, marker_id: int) -> np.ndarray:
        """Get the (4, 4) binary pattern for a given marker ID."""
        return self._patterns[marker_id].copy()

    def generate_marker_image(self, marker_id: int, size: int = 200) -> np.ndarray:
        """Render a marker image using OpenCV."""
        return aruco.generateImageMarker(self._aruco_dict, marker_id, size)
