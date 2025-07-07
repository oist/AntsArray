#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global homography refinement for a 5×5 camera grid with 8-connected overlap.

Input  : EXP_aruco_panorama.pkl  (Frame, Instance, Cam, X, Y)
Output : refined_H_mats.npz  (array H[25,3,3])  +  refined_H_mats.json

Author  : Sam Reiter  •  July 2025
"""

from __future__ import annotations
import argparse, json, time, math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import scipy.io
from scipy.optimize import least_squares


# ──────────────────────────── neighbourhood ──────────────────────────────── #

def make_neighbors_5x5() -> Dict[int, List[int]]:
    """
    Explicit 8-connected neighbour map, hand-coded for clarity.
    """
    nbr = {
        0:  [1,5,6],                 1:  [0,2,5,6,7],     2:  [1,3,6,7,8],
        3:  [2,4,7,8,9],             4:  [3,8,9],
        5:  [0,1,6,10,11],           6:  [0,1,2,5,7,10,11,12],
        7:  [1,2,3,6,8,11,12,13],    8:  [2,3,4,7,9,12,13,14],
        9:  [3,4,8,13,14],
        10: [5,6,11,15,16],          11: [5,6,7,10,12,15,16,17],
        12: [6,7,8,11,13,16,17,18],  13: [7,8,9,12,14,17,18,19],
        14: [8,9,13,18,19],
        15: [10,11,16,20,21],        16: [10,11,12,15,17,20,21,22],
        17: [11,12,13,16,18,21,22,23],18: [12,13,14,17,19,22,23,24],
        19: [13,14,18,23,24],
        20: [15,16,21],              21: [15,16,17,20,22],
        22: [16,17,18,21,23],        23: [17,18,19,22,24],
        24: [18,19,23],
    }
    return nbr


# ───────────────────────────── utilities ─────────────────────────────────── #

# --- parameter ↔ transform ----------------------------------------------
def param_to_RT(p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    θ, tx, ty = p3
    c, s = math.cos(θ), math.sin(θ)
    R = np.array([[c, -s],
                  [s,  c]])
    t = np.array([tx, ty])
    return R, t

def RT_to_param(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    θ = math.atan2(R[1,0], R[0,0])
    return np.array([θ, t[0], t[1]])

# ───────────────────── rigid-body (rotation+translation) fit ────────────── #
def rigid_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch algorithm (2-D, unit scale).
    Returns R (2×2) and t (2,) such that  dst ≈ R·src + t.
    """
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    X = src - c_src
    Y = dst - c_dst
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:          # enforce proper rotation
        Vt[1] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    return R, t


# ────────────────── build pairwise matches for neighbour cams ───────────── #
def pairwise_matches(
        df: pd.DataFrame,
        neighbors: Dict[int, List[int]],
        min_inliers: int = 6,
        ransac_thr: float = 4.0
) -> Dict[Tuple[int, int], Dict]:
    """
    For each neighbouring pair (i,j) find inlier correspondences via RANSAC,
    then fit a *rigid-body* transform (R, t).  Returns::

        { (i,j): {'R':R, 't':t, 'src':src_pts, 'dst':dst_pts}, ... }
    """
    # Pre-index by camera → MultiIndex (Frame,Instance)
    tables = {cam: g.set_index(["Frame", "Instance"])[["X", "Y"]]
              for cam, g in df.groupby("Cam")}

    matches = {}
    for i, nbrs in neighbors.items():
        if i not in tables:
            continue
        tbl_i = tables[i]

        for j in nbrs:
            if j <= i or j not in tables:        # avoid duplicates & missing cams
                continue
            tbl_j = tables[j]

            joined = tbl_i.join(tbl_j, how="inner", lsuffix="_i", rsuffix="_j")
            if len(joined) < min_inliers:
                continue

            src = joined[["X_i", "Y_i"]].to_numpy(float)
            dst = joined[["X_j", "Y_j"]].to_numpy(float)

            # ----- RANSAC just to pick inliers; use homography as a mask -------
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr)
            if mask is None or mask.sum() < min_inliers:
                continue
            inliers = mask.ravel().astype(bool)
            src_in, dst_in = src[inliers], dst[inliers]

            # ----- rigid-body fit on inliers -----------------------------------
            R, t = rigid_transform(src_in, dst_in)
            matches[(i, j)] = dict(R=R, t=t, src=src_in, dst=dst_in)

            print(f"Cam{i:02d} ↔ Cam{j:02d}: {inliers.sum():>4} inliers")

    return matches

# ───────────────────── global optimisation cost ──────────────────────────── #

def build_residual_fun(matches, ref_cam, n_cam):
    # parameter index map
    idx = {}; off = 0
    for c in range(n_cam):
        if c == ref_cam: continue
        idx[c] = slice(off, off+3); off += 3

    # pack pair data
    data = [(i,j,m["src"], m["dst"]) for (i,j), m in matches.items()]

    def theta_to_RT(theta):
        R, t = [], []
        for cam in range(n_cam):
            if cam == ref_cam:
                R.append(np.eye(2));   t.append(np.zeros(2))
            else:
                Ri, ti = param_to_RT(theta[idx[cam]])
                R.append(Ri);          t.append(ti)
        return R, t

    def residual(theta):
        R, t = theta_to_RT(theta)
        errs = []
        for i, j, src, dst in data:
            pi = (R[i] @ src.T).T + t[i]
            pj = (R[j] @ dst.T).T + t[j]
            errs.append((pi - pj).ravel())
        return np.concatenate(errs)

    return residual, idx



# ────────────────────────── main procedure ───────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panorama_pkl", required=True,
                    help="e.g. EXP_aruco_panorama.pkl")
    ap.add_argument("--hmats_init", required=True,
                    help=".mat with `paras` OR .npz with key `H` (zero-based)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--ref_cam", type=int, default=12,
                    help="Camera anchored to identity")
    ap.add_argument("--max_iter", type=int, default=200)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)

    df = pd.read_pickle(args.panorama_pkl)
    needed = {"Frame","Instance","Cam","X","Y"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Pickle missing {needed - set(df.columns)}")

    # load initial per-camera H list (identity if none)
    init_file = Path(args.hmats_init)
    if init_file.suffix == ".npz":
        init_H = list(np.load(init_file)["H"])
    else:                                  # MATLAB
        paras = np.squeeze(scipy.io.loadmat(init_file)["paras"])
        H_pair = [[np.eye(3) if i==j else None for j in range(25)] for i in range(25)]
        for i in range(1,25):
            p = paras[4*(i-1):4*i]
            S = np.array([[p[0],p[1],p[2]],
                          [p[1],p[0],p[3]],
                          [0,0,1.]], float)
            H_pair[0][i] = S
            H_pair[i][0] = np.linalg.inv(S)
        for i in range(1,25):
            for j in range(i+1,25):
                H_pair[i][j] = H_pair[0][j] @ H_pair[i][0]
                H_pair[j][i] = np.linalg.inv(H_pair[i][j])
        init_H = [H_pair[args.ref_cam][i] for i in range(25)]

    # ------- pairwise matches ---------------------------------------------
    neighbors = make_neighbors_5x5()
    matches   = pairwise_matches(df, neighbors)

    # ------- global optimisation ------------------------------------------
    n_cam   = 25
    residual, idx_map = build_residual_fun(matches, args.ref_cam, n_cam)

    theta0 = np.concatenate([RT_to_param(np.eye(2), np.zeros(2))
                         for c in range(n_cam) if c != args.ref_cam])

    print(f"\nOptimising {len(theta0)} parameters "
          f"over {len(matches)} neighbour pairs …")

    res = least_squares(residual, theta0, method="lm", max_nfev=args.max_iter)

    if not res.success:
        print("⚠ optimiser did not fully converge:", res.message)

    # reassemble H list
    # ── helper: convert 2-D rigid transform to 3×3 homogeneous ────────────────
    def RT_to_H(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        R : 2×2 rotation;  t : (2,) translation
        Return homogeneous 3×3 matrix:
            [ R11  R12  tx ]
            [ R21  R22  ty ]
            [  0    0    1 ]
        """
        return np.array([[R[0, 0], R[0, 1], t[0]],
                        [R[1, 0], R[1, 1], t[1]],
                        [0.0,     0.0,     1.0]])

# ── re-assemble one 3×3 H per camera ──────────────────────────────────────
    final_H: list[np.ndarray] = []
    for cam in range(n_cam):
        if cam == args.ref_cam:
            final_H.append(np.eye(3))
        else:
            θ, tx, ty = res.x[idx_map[cam]]
            c, s = math.cos(θ), math.sin(θ)
            R = np.array([[c, -s],
                        [s,  c]])
            t = np.array([tx, ty])
            final_H.append(RT_to_H(R, t))


    # ------- save ----------------------------------------------------------
    np.savez_compressed(outdir/"refined_H_mats.npz", H=np.stack(final_H))
    with (outdir/"refined_H_mats.json").open("w") as f:
        json.dump(dict(time=time.strftime("%F %T"),
                       source=args.panorama_pkl,
                       ref_cam=args.ref_cam,
                       iterations=res.nfev,
                       rms_error=float(np.sqrt((res.fun**2).mean()))),
                  f, indent=2)

    print(f"\n✔ refined_H_mats.npz written to {outdir.resolve()}")
    print(f"   RMS reprojection error: {math.sqrt((res.fun**2).mean()):.3f} px")


if __name__ == "__main__":
    main()
