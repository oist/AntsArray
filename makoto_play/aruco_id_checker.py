"""Simple GUI to read ArUco marker IDs from ant photos.

Pick custom dictionary A (custom_4x4_A100) or B (custom_4x4_B300), open one or
more images, and the detected marker ID(s) are drawn on the image and shown in
big text. Detector settings mirror run_aruco.py's DetectorConfig defaults so the
IDs match what the detection pipeline would report.

Run:
    .venv/Scripts/python.exe makoto_play/aruco_id_checker.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from cv2 import aruco
from PIL import Image, ImageTk

# --- Detector settings (mirrors run_aruco.py DetectorConfig defaults) --------
CORNER_REFINEMENT = "contour"
ADAPTIVE_THRESH_CONSTANT = 3.0
ADAPTIVE_THRESH_WIN_MIN = 10
ADAPTIVE_THRESH_WIN_MAX = 40
ADAPTIVE_THRESH_WIN_STEP = 10
ERROR_CORRECTION_RATE = 1.0
MIN_MARKER_PERIMETER_RATE = 0.03
MAX_MARKER_PERIMETER_RATE = 4.0
POLYGONAL_APPROX_ACCURACY_RATE = 0.03

REPO_ROOT = Path(__file__).resolve().parent.parent
DICT_DIR = REPO_ROOT / "aruco_detection" / "custom_dicts"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_CANVAS = (1000, 750)  # displayed image is scaled to fit within this box


def imread_unicode(path: Path) -> np.ndarray | None:
    """Read an image, tolerating non-ASCII paths that cv2.imread mishandles."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def load_custom_aruco_dict(npz_path: Path) -> aruco.Dictionary:
    """Load a custom .npz dictionary (same logic as run_aruco.py)."""
    data = np.load(str(npz_path), allow_pickle=True)
    if "bytesList" not in data.files:
        raise ValueError(f"Custom ArUco dictionary missing bytesList: {npz_path}")
    if "max_correction_bits" not in data.files:
        raise ValueError(f"Custom ArUco dictionary missing max_correction_bits: {npz_path}")
    custom = aruco.Dictionary()
    custom.bytesList = data["bytesList"]
    custom.markerSize = int(data["marker_size"]) if "marker_size" in data.files else 4
    custom.maxCorrectionBits = int(data["max_correction_bits"])
    return custom


def find_dictionaries() -> dict[str, Path]:
    """Map 'A'/'B' to the newest matching .npz in the custom_dicts folder."""
    found: dict[str, Path] = {}
    for label, prefix in (("A", "custom_4x4_A"), ("B", "custom_4x4_B")):
        matches = sorted(DICT_DIR.glob(f"{prefix}*.npz"))
        if matches:
            found[label] = matches[-1]  # newest by name (timestamped suffix)
    return found


def _corner_refine_enum(name: str) -> int:
    attr = {
        "none": "CORNER_REFINE_NONE",
        "subpix": "CORNER_REFINE_SUBPIX",
        "contour": "CORNER_REFINE_CONTOUR",
        "apriltag": "CORNER_REFINE_APRILTAG",
    }.get(name.lower(), "CORNER_REFINE_CONTOUR")
    return int(getattr(aruco, attr, aruco.CORNER_REFINE_CONTOUR))


def build_detector(dictionary: aruco.Dictionary) -> aruco.ArucoDetector:
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = _corner_refine_enum(CORNER_REFINEMENT)
    params.adaptiveThreshConstant = ADAPTIVE_THRESH_CONSTANT
    params.adaptiveThreshWinSizeMin = ADAPTIVE_THRESH_WIN_MIN
    params.adaptiveThreshWinSizeMax = ADAPTIVE_THRESH_WIN_MAX
    params.adaptiveThreshWinSizeStep = ADAPTIVE_THRESH_WIN_STEP
    params.errorCorrectionRate = ERROR_CORRECTION_RATE
    params.minMarkerPerimeterRate = MIN_MARKER_PERIMETER_RATE
    params.maxMarkerPerimeterRate = MAX_MARKER_PERIMETER_RATE
    params.polygonalApproxAccuracyRate = POLYGONAL_APPROX_ACCURACY_RATE
    return aruco.ArucoDetector(dictionary, params)


class ArucoIdChecker(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ArUco ID Checker")
        self.geometry("1120x900")

        self.dictionaries = find_dictionaries()
        self.detectors: dict[str, aruco.ArucoDetector] = {}
        self.image_paths: list[Path] = []
        self.index = 0
        self._photo: ImageTk.PhotoImage | None = None  # keep a reference alive

        self.selected_dict = tk.StringVar(value="A" if "A" in self.dictionaries else "B")

        self._build_controls()
        self._build_canvas()
        self._build_result_bar()

        if not self.dictionaries:
            messagebox.showerror(
                "No dictionaries found",
                f"No custom_4x4_A*/B*.npz files found in:\n{DICT_DIR}",
            )

    # -- UI construction ------------------------------------------------------
    def _build_controls(self) -> None:
        bar = tk.Frame(self, pady=8, padx=8)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text="Dictionary:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        for label in ("A", "B"):
            state = tk.NORMAL if label in self.dictionaries else tk.DISABLED
            text = label
            if label in self.dictionaries:
                text = f"{label}  ({self.dictionaries[label].stem})"
            tk.Radiobutton(
                bar, text=text, value=label, variable=self.selected_dict,
                state=state, command=self._redetect, font=("Segoe UI", 10),
            ).pack(side=tk.LEFT, padx=4)

        tk.Frame(bar, width=20).pack(side=tk.LEFT)
        tk.Button(bar, text="Open image(s)...", command=self.open_images).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Open folder...", command=self.open_folder).pack(side=tk.LEFT, padx=4)

        nav = tk.Frame(self, padx=8)
        nav.pack(side=tk.TOP, fill=tk.X)
        tk.Button(nav, text="< Prev", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(nav, text="Next >", command=self.next_image).pack(side=tk.LEFT, padx=4)
        self.counter_label = tk.Label(nav, text="no image loaded", font=("Segoe UI", 9))
        self.counter_label.pack(side=tk.LEFT, padx=10)

    def _build_canvas(self) -> None:
        self.canvas = tk.Canvas(self, bg="#222222", width=MAX_CANVAS[0], height=MAX_CANVAS[1])
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_result_bar(self) -> None:
        self.result_label = tk.Label(
            self, text="Open an image to read its marker ID.",
            font=("Segoe UI", 20, "bold"), fg="#0a7d00", pady=10,
        )
        self.result_label.pack(side=tk.BOTTOM, fill=tk.X)

    # -- Detector cache -------------------------------------------------------
    def _get_detector(self) -> aruco.ArucoDetector | None:
        label = self.selected_dict.get()
        if label not in self.dictionaries:
            return None
        if label not in self.detectors:
            self.detectors[label] = build_detector(load_custom_aruco_dict(self.dictionaries[label]))
        return self.detectors[label]

    # -- Image loading & navigation -------------------------------------------
    def open_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select image(s)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if paths:
            self.image_paths = [Path(p) for p in paths]
            self.index = 0
            self._show_current()

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select a folder of images")
        if not folder:
            return
        paths = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            messagebox.showinfo("No images", f"No images found in:\n{folder}")
            return
        self.image_paths = paths
        self.index = 0
        self._show_current()

    def next_image(self) -> None:
        if self.image_paths:
            self.index = (self.index + 1) % len(self.image_paths)
            self._show_current()

    def prev_image(self) -> None:
        if self.image_paths:
            self.index = (self.index - 1) % len(self.image_paths)
            self._show_current()

    def _redetect(self) -> None:
        if self.image_paths:
            self._show_current()

    # -- Core: detect + render ------------------------------------------------
    def _show_current(self) -> None:
        path = self.image_paths[self.index]
        self.counter_label.config(text=f"[{self.index + 1}/{len(self.image_paths)}]  {path.name}")

        image = imread_unicode(path)
        if image is None:
            self.result_label.config(text=f"Could not read image: {path.name}", fg="#b00000")
            self.canvas.delete("all")
            return

        detector = self._get_detector()
        if detector is None:
            self.result_label.config(text="No dictionary selected/available.", fg="#b00000")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        annotated = image.copy()
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(annotated, corners, ids, borderColor=(0, 255, 0))
            id_list = [int(i) for i in ids.flatten()]
            self._draw_id_labels(annotated, corners, id_list)
            label = self.selected_dict.get()
            joined = ", ".join(str(i) for i in id_list)
            noun = "ID" if len(id_list) == 1 else "IDs"
            self.result_label.config(text=f"Dict {label}  ->  {noun}: {joined}", fg="#0a7d00")
        else:
            label = self.selected_dict.get()
            self.result_label.config(
                text=f"Dict {label}  ->  no marker found (try the other dictionary)",
                fg="#b00000",
            )

        self._render(annotated)

    def _draw_id_labels(self, image: np.ndarray, corners, id_list: list[int]) -> None:
        """Draw a big ID number near each detected marker."""
        scale = max(image.shape[1] / 1000.0, 1.0)
        font_scale = 1.2 * scale
        thickness = max(2, int(2 * scale))
        for marker_corners, marker_id in zip(corners, id_list):
            pts = marker_corners.reshape(-1, 2)
            cx, cy = pts.mean(axis=0)
            org = (int(cx) + int(15 * scale), int(cy) - int(15 * scale))
            text = str(marker_id)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 0), thickness, cv2.LINE_AA)

    def _render(self, bgr_image: np.ndarray) -> None:
        h, w = bgr_image.shape[:2]
        scale = min(MAX_CANVAS[0] / w, MAX_CANVAS[1] / h, 1.0)
        disp = cv2.resize(bgr_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or MAX_CANVAS[0]
        ch = self.canvas.winfo_height() or MAX_CANVAS[1]
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)


def main() -> None:
    app = ArucoIdChecker()
    app.mainloop()


if __name__ == "__main__":
    main()
