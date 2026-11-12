#!/usr/bin/env python3
"""Inspect an ant cup's raw videos with synchronized original sleep labels.

python -m tracking.single_ant.video_viewer DATASET --cup 30
Requires PyAV, OpenCV, Pillow and Tk. Uses original full-posture S0/S1 only;
raw ArUco/SLEAP and assigned-track overlays. The abandoned partial-posture
candidate is never read.
"""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time

import numpy as np
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracking.single_ant.cup_video import CameraDecoder, Cancelled
from tracking.single_ant.state_overlay import intervals

COLORS = {-1: '#a8adb7', 0: '#d99728', 1: '#8452b8'}


def mounted_path(path, root):
    path, root = Path(path), Path(root)
    if path.exists():
        return path
    if '/bucket/' in str(root) and '/bucket/' in str(path):
        candidate = Path(str(root).split('/bucket/', 1)[0] + '/bucket/' + str(path).split('/bucket/', 1)[1])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def segment_at(segments, frame):
    for i, segment in enumerate(segments):
        lo = int(segment['frame_offset'])
        if lo <= frame < lo + int(segment['num_frames']):
            return i, frame - lo
    return None, None


def full_posture_categories(observed):
    category = np.full(len(observed), -1, np.int8)
    known = observed >= 0
    category[known] = (observed[known] <= 1).astype(np.int8)
    return category


def cup_maps(cup, matrices, mm_per_pixel, size, width=4024, height=3036):
    """Maps camera pixels into a cup-only panorama; calibration has its own scale."""
    import cv2
    radius = float(cup['radius_mm']) * 1.06
    center = np.asarray(cup['center_mm'], float)
    axis = (np.arange(size) + .5) / size * 2 * radius - radius
    xx, yy = np.meshgrid(axis + center[0], axis + center[1])
    board = np.stack([xx / mm_per_pixel, yy / mm_per_pixel], axis=2)
    disk = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= cup['radius_mm'] ** 2
    plans = []
    for camera, matrix in enumerate(matrices):
        source = cv2.perspectiveTransform(board, np.linalg.inv(matrix))
        x, y = source[:, :, 0], source[:, :, 1]
        valid = (x >= 1) & (x < width - 2) & (y >= 1) & (y < height - 2)
        coverage = float(valid[disk].mean())
        if coverage < .015:
            continue
        weight = np.maximum(0, np.minimum.reduce([x, width - 1 - x, y, height - 1 - y]))
        plans.append(dict(camera=camera, map_x=x.astype(np.float32), map_y=y.astype(np.float32),
                          valid=valid, weight=weight.astype(np.float32), coverage=coverage))
    if not plans:
        raise ValueError('No calibrated camera covers this cup')
    # Use a single camera when it covers the cup; otherwise cover the cup greedily.
    ordered = sorted(plans, key=lambda p: p['coverage'], reverse=True)
    needed, covered = [], np.zeros_like(disk)
    for _ in range(len(plans)):
        best = max((p for p in plans if p['camera'] not in needed),
                   key=lambda p: int((p['valid'] & disk & ~covered).sum()))
        needed.append(best['camera']);covered |= best['valid']
        if covered[disk].mean() >= .999:
            break
    return plans, needed, dict(center=center, radius=radius, size=size)


class CupData:
    def __init__(self, path, size=1000):
        self.root = Path(path).expanduser().resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.fps = float(self.manifest['fps']);self.n = int(self.manifest['n_frames'])
        self.origin = datetime.fromisoformat(self.manifest['start_datetime'])
        self.segments = self.manifest.get('recording_segments') or [dict(frame_offset=0, num_frames=self.n, source_block=self.manifest['source_block'])]
        self.cups = sorted(self.manifest.get('physical_cups', self.manifest['cups']), key=lambda c: c['cup_id'])
        meta = json.loads((self.root / 'panorama_metadata.json').read_text())
        self.mm_per_pixel = float(meta['mm_per_pixel'])
        self.matrices = np.load(mounted_path(meta['homographies'], self.root))['H']
        with (self.root / 'cup_identities.csv').open(newline='') as file:
            identities = {int(row['CupID']): row for row in csv.DictReader(file)}
        metadata = self.root / 'analysis/speed_by_metadata/per_ant_summary.csv'
        self.metadata = {}
        if metadata.exists():
            with metadata.open(newline='') as file:
                self.metadata = {int(row['CupID']): row for row in csv.DictReader(file)}
        self.choices = []
        for cup in self.cups:
            cid = int(cup['cup_id']);identity = identities.get(cid, {})
            item = self.metadata.get(cid, {})
            tid = int(identity.get('TrackID') or cid) if identity else None
            tag = item.get('ExpectedID') or identity.get('ArucoID') or 'unassigned'
            self.choices.append(dict(cup=cid, track=tid, tag=tag,
                label=f"Cup {cid:02d} · ID {tag} · {item.get('Colony', '')} · {item.get('Role', '')}",
                identity_status=item.get('IdentityStatus', 'matched' if identity else 'no track')))
        self.size = int(size)
        self.use_video_motion = False
        self.video_motion_root = self.root / "analysis/sleep_s01_video_energy"
        self.state_cache = OrderedDict();self.geometry_cache = {};self.video_cache = {}

    def labels(self, index):
        track = self.choices[index]['track']
        key = (track, bool(getattr(self, 'use_video_motion', False)))
        if key not in self.state_cache:
            observed = np.full(self.n, -1, np.int8)
            if track is not None:
                path = self.root / 'analysis/sleep_s01_exclusion/per_track' / f'TrackID_{track:04d}_all_single' / 'labels.npz'
                if path.exists():
                    # Explicitly use the original state assignments. Ignore both
                    # the partial classifier and motion-filled category labels.
                    with np.load(path) as data:
                        observed = data['observed']
                else:
                    from tracking.single_ant.sleep_states import observed_states
                    eigen = self.root / 'analysis/eigenposture'
                    inventory = eigen / 'inventory_all.csv'
                    if inventory.exists():
                        with inventory.open(newline='') as file:
                            rows = list(csv.DictReader(file))
                        selected = next((r for r in rows if int(r['track_id']) == track), None)
                        if selected:
                            observed = observed_states(eigen, selected['ant'], self.n, selected['selected'].lower() == 'true')
            if observed.shape != (self.n,):
                raise ValueError('Full-posture labels have the wrong frame span')
            recorded = np.zeros(self.n, bool)
            for segment in self.segments:
                lo = int(segment['frame_offset']);recorded[lo:lo + int(segment['num_frames'])] = True
            observed[~recorded] = -1
            if self.choices[index]['identity_status'] != 'matched':
                observed[:] = -1
            categories = full_posture_categories(observed)
            video_motion = np.zeros(self.n, bool)
            if getattr(self, 'use_video_motion', False):
                root = self.video_motion_root
                cup = self.choices[index]['cup']
                path = root / 'per_cup' / f'cup_{cup:04d}' / 'labels.npz'
                release = root / 'application.json'
                if path.exists() and release.exists() and json.loads(release.read_text()).get('released'):
                    with np.load(path) as derived:
                        if derived['category'].shape != (self.n,):
                            raise ValueError('Cup motion labels have the wrong frame span')
                        video_motion = (observed < 0) & (derived['category'] == 0) & (derived['evidence'] == 32) & recorded
                    categories[video_motion] = 0
            self.state_cache[key] = dict(observed=observed, category=categories, video_motion=video_motion,
                changes=np.flatnonzero(np.diff(categories) != 0) + 1, sleep=intervals(categories == 1))
            while len(self.state_cache) > 4:
                self.state_cache.popitem(last=False)
        return self.state_cache[key]

    def geometry(self, index):
        if index not in self.geometry_cache:
            self.geometry_cache[index] = cup_maps(self.cups[index], self.matrices, self.mm_per_pixel, self.size)
        return self.geometry_cache[index]

    def video(self, segment_index, camera):
        key = segment_index, camera
        if key not in self.video_cache:
            segment = self.segments[segment_index]
            block = mounted_path(segment['source_block'], self.root)
            videos = list(block.glob(f'cam{camera + 1:02d}_*.mkv'))
            if len(videos) != 1:
                raise ValueError(f'Expected one source video for cam{camera + 1:02d} in {block}')
            path = videos[0]
            diagnostic = json.loads(path.with_suffix(path.suffix + '.diag.json').read_text())
            provenance = diagnostic['provenance'];first = provenance['firstEncodedFrame'];last = provenance['lastEncodedFrame']
            n = int(segment['num_frames'])
            if (first['blockId'] != 0 or first['fileFrameIndex'] != 0 or last['blockId'] != n - 1
                    or last['fileFrameIndex'] != n - 1 or any(provenance.get(k) for k in ['gaps', 'resetEvents', 'invalidIdEvents', 'wrapEvents'])
                    or provenance.get('eventOverflowCount', 0)):
                raise ValueError('Source video lacks the verified common-trigger frame alignment')
            if float(diagnostic['context']['fps']) != self.fps:
                raise ValueError('Source video FPS does not match tracking')
            self.video_cache[key] = path
        return self.video_cache[key]

    def status(self, index, frame):
        labels = self.labels(index)
        state = int(labels['observed'][frame]);category = int(labels['category'][frame])
        segment, local = segment_at(self.segments, frame)
        if segment is None:
            return -1, 'NOT RECORDED', 'No video during the gap between blocks'
        if state < 0 and category == 0:
            return 0, 'NOT S0/S1 · MOTION', 'High cup-video motion; original full posture ambiguous. Low motion never adds sleep.'
        if state < 0:
            return -1, 'AMBIGUOUS', 'Full posture unavailable' if self.choices[index]['identity_status'] == 'matched' else self.choices[index]['identity_status']
        return category, 'SLEEP · S0/S1' if category == 1 else 'NOT S0/S1', f'Original full-posture state S{state}'


class CupRenderer:
    def __init__(self, data):
        self.data = data;self.key = None;self.readers = {};self.cache = OrderedDict()
        self.cache_limit = max(16, int(512 * 1024 ** 2 / data.size ** 2))
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='cup-camera')

    def close_readers(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def close(self):
        self.close_readers();self.pool.shutdown(wait=False, cancel_futures=True)

    def render(self, index, frame, camera=None, cancelled=lambda: False):
        import cv2
        segment, local = segment_at(self.data.segments, frame)
        if segment is None:
            return np.full((self.data.size, self.data.size), 85, np.uint8), []
        key = index, segment
        if key != self.key:
            self.close_readers();self.cache.clear();self.key = key
        cache_key = frame, camera
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        plans, needed, geometry = self.data.geometry(index)
        plans = [p for p in plans if p['camera'] in needed] if camera is None else [p for p in plans if p['camera'] == camera]
        for plan in plans:
            cam = plan['camera']
            if cam not in self.readers:
                self.readers[cam] = CameraDecoder(self.data.video(segment, cam), self.data.fps,
                    self.data.segments[segment]['num_frames'], plan['map_x'], plan['map_y'])
        jobs = [self.pool.submit(self.readers[p['camera']].read, local, cancelled) for p in plans]
        patches = []
        failure = None
        for job in jobs:
            try:
                patches.append(job.result())
            except Exception as error:
                failure = error
        if failure:
            raise failure
        if cancelled():
            raise Cancelled
        weights = np.stack([p['weight'] for p in plans])
        winner = weights.argmax(axis=0)
        result = np.full((self.data.size, self.data.size), 75, np.uint8)
        for i, patch in enumerate(patches):
            mask = (winner == i) & plans[i]['valid']
            result[mask] = patch[mask]
        # A subtle cup boundary, with no tracking overlay obscuring the animal.
        cv2.circle(result, (self.data.size // 2, self.data.size // 2), int(self.data.size / 2 / 1.06), 210, 1, cv2.LINE_AA)
        value = result, [p['camera'] + 1 for p in plans]
        self.cache[cache_key] = value
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return value


class RenderWorker:
    def __init__(self, renderer, detections=None):
        self.detections = detections
        self.renderer = renderer;self.requests = queue.Queue();self.results = queue.Queue()
        self.generation = 0;self.stopping = False
        threading.Thread(target=self.run, daemon=True, name='cup-video').start()

    def request(self, index, frame, camera, overlays=None):
        self.generation += 1
        self.requests.put((self.generation, index, int(frame), camera, overlays or {}))
        return self.generation

    def run(self):
        while not self.stopping:
            request = self.requests.get()
            if request is None:
                break
            while not self.requests.empty():
                latest = self.requests.get()
                if latest is None:
                    self.stopping = True;break
                request = latest
            if self.stopping:
                break
            token, index, frame, camera, overlays = request
            try:
                result = self.renderer.render(index, frame, camera, lambda: self.stopping or token != self.generation)
                payload = self.detections.frame(index, frame, camera, overlays, lambda: self.stopping or token != self.generation) if self.detections else None
                self.results.put((token, index, frame, (*result, payload, overlays), None))
            except Cancelled:
                pass
            except Exception as error:
                self.results.put((token, index, frame, None, str(error)))
        self.renderer.close()
        if self.detections:self.detections.close()

    def close(self):
        self.stopping = True;self.generation += 1;self.requests.put(None)


def timeline_image(category, first, stop, width, height=28):
    """Every sample contributes to each pixel's three-color composition."""
    from PIL import Image
    width = max(1, int(width));first = max(0, int(first));stop = min(len(category), int(stop))
    image = np.full((height, width, 3), 168, np.uint8)
    if stop <= first:
        return Image.fromarray(image)
    edges = np.linspace(first, stop, width + 1).astype(int)
    values = np.array([category[first:stop] == c for c in [1, 0, -1]])
    cumulative = np.c_[np.zeros(3, np.int64), values.cumsum(axis=1)]
    counts = cumulative[:, edges[1:] - first] - cumulative[:, edges[:-1] - first]
    total = counts.sum(axis=0)
    palette = np.array([[132, 82, 184], [217, 151, 40], [168, 173, 183]], np.uint8)
    for x in range(width):
        if not total[x]:
            image[:, x] = palette[{1: 0, 0: 1, -1: 2}[int(category[min(edges[x], stop - 1)])]]
            continue
        boundaries = np.rint(np.r_[0, counts[:, x].cumsum()] / total[x] * height).astype(int)
        for i in range(3):
            image[boundaries[i]:boundaries[i + 1], x] = palette[i]
    return Image.fromarray(image)


class CupVideoViewer:
    def __init__(self, root, path, cup=30, start_frame=0, size=1000, cup_motion=False):
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import ttk
        from PIL import Image, ImageTk
        self.tk, self.ttk, self.Image, self.ImageTk = tk, ttk, Image, ImageTk
        # NoMachine reports inconsistent physical DPI; use readable, scalable
        # fonts in this interpreter without changing other open applications.
        root.tk.call('tk', 'scaling', 1.25)
        for name in ['TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont']:
            tkfont.nametofont(name).configure(family='DejaVu Sans', size=11)
        ttk.Style(root).theme_use('clam')
        self.root = root;self.data = CupData(path, size)
        self.data.use_video_motion = bool(cup_motion)
        from tracking.single_ant.cup_detections import DetectionReader
        self.detections = DetectionReader(self.data)
        self.worker = RenderWorker(CupRenderer(self.data), self.detections)
        self.overlay_payload = None;self.rendered_overlay_options = {};self.overlay_picture = None
        self.index = next((i for i, choice in enumerate(self.data.choices) if choice['cup'] == cup), 0)
        self.frame = min(max(0, int(start_frame)), self.data.n - 1)
        self.playing = False;self.direction = 1;self.busy = False;self.pending_token = 0
        self.image = None;self.photo = None;self.zoom = 1.;self.focus = np.array([.5, .5])
        self.draw_box = None;self.drag_origin = None;self.render_count = 0;self.closed = False
        self.local_span = 5 * 60 * self.data.fps;self.local_bounds = (0, 1)
        self.last_local_key = None;self.last_full_width = 0;self.cameras = [];self.request_started = 0.
        self.play_after = None
        root.title('Single-ant cup video · detections and tracking')
        root.geometry('1600x1120+180+70');root.minsize(1050, 800)
        root.setvar('single_ant_cup_video_dataset', str(self.data.root))
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.columnconfigure(0, weight=1);root.rowconfigure(1, weight=1)
        top = ttk.Frame(root, padding=8);top.grid(row=0, column=0, sticky='ew')
        ttk.Button(top, text='Previous cup', command=lambda: self.change_cup(-1)).pack(side='left')
        self.cup_selector = ttk.Combobox(top, state='readonly', width=56, values=[x['label'] for x in self.data.choices])
        self.cup_selector.current(self.index);self.cup_selector.pack(side='left', padx=6)
        self.cup_selector.bind('<<ComboboxSelected>>', lambda e: self.select_cup(self.cup_selector.current()))
        ttk.Button(top, text='Next cup', command=lambda: self.change_cup(1)).pack(side='left')
        self.motion_overlay = tk.BooleanVar(value=bool(cup_motion))
        ttk.Checkbutton(top, text='Cup motion exclusions', variable=self.motion_overlay, command=self.reload_motion).pack(side='right', padx=5)
        ttk.Button(top, text='Reload motion', command=self.reload_motion).pack(side='right', padx=5)
        body = ttk.Frame(root);body.grid(row=1, column=0, sticky='nsew')
        body.columnconfigure(1, weight=1);body.rowconfigure(1, weight=1)
        side = ttk.Frame(body, padding=(10, 5, 10, 5));side.grid(row=0, column=0, rowspan=2, sticky='ns')
        self.overview = tk.Canvas(side, width=285, height=218, bg='white', highlightthickness=0)
        self.overview.pack();self.overview.bind('<Button-1>', self.overview_click)
        ttk.Label(side, text='Click a cup to select it').pack(pady=(2, 14))
        self.badge = tk.Label(side, text='LOADING', font=('DejaVu Sans', 18, 'bold'), bg='#a8adb7', fg='white', padx=12, pady=14, width=19)
        self.badge.pack(fill='x')
        self.state_detail = tk.StringVar();ttk.Label(side, textvariable=self.state_detail, wraplength=285).pack(fill='x', pady=(6, 12))
        self.cup_detail = tk.StringVar();ttk.Label(side, textvariable=self.cup_detail, wraplength=285, justify='left').pack(fill='x')
        ttk.Label(side, text='Camera view').pack(anchor='w', pady=(18, 3))
        self.view = ttk.Combobox(side, state='readonly', width=26)
        self.view.pack(fill='x');self.view.bind('<<ComboboxSelected>>', lambda e: self.seek(self.frame))
        self.view.set('Cup panorama')
        ttk.Button(side, text='Show tracked camera', command=self.track_camera).pack(fill='x', pady=(5, 0))
        ttk.Button(side, text='Fit cup', command=self.fit).pack(fill='x', pady=(16, 3))
        ttk.Button(side, text='Zoom in', command=lambda: self.zoom_by(1.5)).pack(fill='x', pady=3)
        ttk.Button(side, text='Zoom out', command=lambda: self.zoom_by(1 / 1.5)).pack(fill='x', pady=3)
        ttk.Label(side, text='Mouse wheel: zoom\nDrag video: pan\n\nSpace: play / pause\nJ / K / L: rewind / pause / play\n← / →: one frame\nShift + ← / →: five seconds', wraplength=285).pack(anchor='w', pady=18)
        overlay_bar = ttk.Frame(body, padding=(5, 3));overlay_bar.grid(row=0, column=1, sticky='ew')
        self.overlay_vars = {}
        for j, (key, label, enabled) in enumerate([('aruco','ArUco IDs',True),('sleap','Raw SLEAP',True),
                ('track','Tracked pose',True),('trail','5 s track trail',False),('names','Node names',False),
                ('all_cameras','All camera detections',False)]):
            variable = tk.BooleanVar(value=enabled);self.overlay_vars[key] = variable
            ttk.Checkbutton(overlay_bar, text=label, variable=variable, command=self.overlay_changed).grid(row=j//4,column=j%4,sticky='w',padx=5)
        confidence = ttk.Frame(overlay_bar);confidence.grid(row=1,column=2,columnspan=2,sticky='w')
        ttk.Label(confidence,text='Raw node min score:').pack(side='left')
        self.node_score = tk.StringVar(value='0.00')
        score = ttk.Spinbox(confidence,from_=0,to=1,increment=.05,width=5,textvariable=self.node_score,command=self.overlay_changed)
        score.pack(side='left',padx=4);score.bind('<Return>',lambda event:self.overlay_changed());score.bind('<FocusOut>',lambda event:self.overlay_changed())
        self.overlay_detail = tk.StringVar(value='Loading raw detections and assigned track…')
        ttk.Label(overlay_bar,text='Green: ArUco · magenta: raw SLEAP · cyan: track · yellow: position/trail · hollow raw nodes: score <0.2').grid(row=2,column=0,columnspan=4,sticky='w',padx=5)
        ttk.Label(overlay_bar,textvariable=self.overlay_detail,wraplength=1150).grid(row=3,column=0,columnspan=4,sticky='w',padx=5)
        self.canvas = tk.Canvas(body, bg='#18191d', highlightthickness=0)
        self.canvas.grid(row=1, column=1, sticky='nsew')
        self.canvas.bind('<Configure>', lambda e: self.draw_video())
        self.canvas.bind('<MouseWheel>', lambda e: self.zoom_by(1.25 if e.delta > 0 else .8, e))
        self.canvas.bind('<Button-4>', lambda e: self.zoom_by(1.25, e));self.canvas.bind('<Button-5>', lambda e: self.zoom_by(.8, e))
        self.canvas.bind('<ButtonPress-1>', self.pan_start);self.canvas.bind('<B1-Motion>', self.pan_drag)
        controls = ttk.Frame(root, padding=(8, 6));controls.grid(row=2, column=0, sticky='ew')
        for label, command in [('−60 s', lambda: self.jump(-60)), ('−5 s', lambda: self.jump(-5)), ('◀ Frame', lambda: self.step(-1)),
                               ('Rewind', self.rewind), ('Pause', self.pause), ('Play', self.play), ('Fast forward', self.fast_forward),
                               ('Frame ▶', lambda: self.step(1)), ('+5 s', lambda: self.jump(5)), ('+60 s', lambda: self.jump(60))]:
            ttk.Button(controls, text=label, command=command).pack(side='left', padx=2)
        self.rate = tk.StringVar(value='1×')
        ttk.Combobox(controls, state='readonly', textvariable=self.rate, width=6,
                     values=['0.1×', '0.25×', '0.5×', '1×', '2×', '4×', '8×', '16×', '32×']).pack(side='left', padx=(12, 2))
        self.play_status = tk.StringVar(value='Paused');ttk.Label(controls, textvariable=self.play_status).pack(side='left', padx=10)
        navigation = ttk.Frame(root, padding=(8, 0, 8, 6));navigation.grid(row=3, column=0, sticky='ew')
        ttk.Label(navigation, text='Go to hours:').pack(side='left')
        self.hour = tk.StringVar(value=f'{self.frame / self.data.fps / 3600:.6f}')
        entry = ttk.Entry(navigation, textvariable=self.hour, width=12);entry.pack(side='left', padx=5)
        entry.bind('<Return>', lambda e: self.goto_hour())
        ttk.Button(navigation, text='Go', command=self.goto_hour).pack(side='left')
        ttk.Button(navigation, text='Previous label change', command=lambda: self.label_change(-1)).pack(side='left', padx=(14, 3))
        ttk.Button(navigation, text='Next label change', command=lambda: self.label_change(1)).pack(side='left', padx=3)
        ttk.Button(navigation, text='Previous sleep', command=lambda: self.sleep_jump(-1)).pack(side='left', padx=(14, 3))
        ttk.Button(navigation, text='Next sleep', command=lambda: self.sleep_jump(1)).pack(side='left', padx=3)
        ttk.Label(navigation, text='Local timeline:').pack(side='left', padx=(16, 4))
        self.span = ttk.Combobox(navigation, state='readonly', width=7, values=['1 min', '5 min', '15 min', '60 min'])
        self.span.set('5 min');self.span.pack(side='left');self.span.bind('<<ComboboxSelected>>', self.set_span)
        timelines = ttk.Frame(root, padding=(10, 0, 10, 0));timelines.grid(row=4, column=0, sticky='ew');timelines.columnconfigure(0, weight=1)
        self.full_timeline = tk.Canvas(timelines, height=31, highlightthickness=0)
        self.full_timeline.grid(row=0, column=0, sticky='ew');self.full_timeline.bind('<Configure>', lambda e: self.draw_timelines(True))
        self.full_timeline.bind('<Button-1>', lambda e: self.seek(round(e.x / max(1, self.full_timeline.winfo_width()) * (self.data.n - 1))))
        self.timeline_text = tk.StringVar();ttk.Label(timelines, textvariable=self.timeline_text).grid(row=1, column=0, sticky='w')
        self.local_timeline = tk.Canvas(timelines, height=39, highlightthickness=0);self.local_timeline.grid(row=2, column=0, sticky='ew')
        self.local_timeline.bind('<Button-1>', lambda e: self.seek(round(self.local_bounds[0] + e.x / max(1, self.local_timeline.winfo_width()) * (self.local_bounds[1] - self.local_bounds[0]))))
        self.clock = tk.StringVar();ttk.Label(root, textvariable=self.clock, padding=(10, 5)).grid(row=5, column=0, sticky='ew')
        self.message = tk.StringVar();ttk.Label(root, textvariable=self.message, padding=(10, 0, 10, 8), wraplength=1500).grid(row=6, column=0, sticky='ew')
        for key, action in [('<space>', self.toggle), ('j', self.rewind), ('k', self.pause), ('l', self.play),
                            ('<Left>', lambda: self.step(-1)), ('<Right>', lambda: self.step(1)),
                            ('<Shift-Left>', lambda: self.jump(-5)), ('<Shift-Right>', lambda: self.jump(5))]:
            root.bind(key, lambda event, command=action: self.shortcut(event, command))
        self.prepare_cup();root.after(25, self.poll);root.after(100, lambda: self.seek(self.frame))

    def overlay_options(self):
        options = {key: variable.get() for key,variable in self.overlay_vars.items()}
        try:options['min_score'] = float(np.clip(float(self.node_score.get()),0,1))
        except ValueError:options['min_score'] = 0.
        return options

    def overlay_changed(self):
        # Request the current frame without changing playback state. Cached
        # video pixels are reused; the worker returns overlays for that frame.
        if self.image is not None:self.request(self.frame)

    def track_camera(self):
        if not (self.overlay_vars['track'].get() or self.overlay_vars['trail'].get()):
            self.message.set('Enable Tracked pose to select its source camera.');return
        if not self.overlay_payload or not self.overlay_payload.get('track'):
            self.message.set('No assigned track at this frame.');return
        track = self.overlay_payload['track']
        camera = track['camera']
        if camera is None or camera < 0:camera = track['aruco_camera']
        if camera not in self.cameras:
            self.message.set('The track has no source camera covering this cup.');return
        self.view.current(self.cameras.index(camera)+1);self.request(self.frame)

    def reload_motion(self):
        self.data.use_video_motion = bool(self.motion_overlay.get())
        self.data.state_cache.clear()
        self.labels = self.data.labels(self.index)
        self.last_full_width = 0;self.last_local_key = None
        category, text, detail = self.data.status(self.index, self.frame)
        self.badge.configure(text=text, bg='#3f9a79' if self.labels['video_motion'][self.frame] else COLORS[category])
        self.state_detail.set(detail)
        self.draw_timelines(True)
        root = self.data.video_motion_root
        if not (root / 'application.json').exists():
            self.message.set('Cup motion results are still processing; original full-posture labels remain visible.')
        else:
            release = json.loads((root / 'application.json').read_text())
            assembly = json.loads((root / 'assembly.json').read_text())
            self.message.set(f"Cup motion released: {release['released']}; measured groups {assembly['complete_groups']}/{assembly['expected_groups']}. Sleep always uses original full posture.")

    def shortcut(self, event, command):
        if event.widget.winfo_class() in ['TEntry', 'Entry', 'TCombobox', 'TSpinbox', 'Spinbox']:
            return
        command();return 'break'

    def prepare_cup(self):
        choice = self.data.choices[self.index]
        self.labels = self.data.labels(self.index)
        plans, needed, _ = self.data.geometry(self.index)
        self.cameras = sorted(p['camera'] for p in plans)
        self.view.configure(values=['Cup panorama'] + [f'Camera {c + 1:02d}' for c in self.cameras]);self.view.current(0)
        info = self.data.metadata.get(choice['cup'], {})
        self.cup_detail.set(f"Cup {choice['cup']:02d} · resident ID {choice['tag']}\n"
            f"Colony {info.get('Colony', 'unknown')} · {info.get('Role', 'unknown')}\n"
            + (f"{info.get('IdentityStatus')}\n" if info.get('IdentityStatus') != 'matched' else '')
            + f"Full posture" + (" + cup motion exclusion" if self.data.use_video_motion else " only") + f"\n{self.data.n / self.data.fps / 3600:.2f} h stitched recording")
        self.last_full_width = 0;self.last_local_key = None;self.zoom = 1.;self.focus[:] = .5
        self.draw_overview();self.draw_timelines(True)

    def select_cup(self, index):
        self.pause();self.index = int(index);self.cup_selector.current(self.index)
        self.overlay_detail.set('Loading detections and track…')
        self.image = None;self.overlay_picture = None;self.overlay_payload = None;self.canvas.delete('all');self.badge.configure(text='LOADING', bg='#a8adb7');self.state_detail.set('')
        self.prepare_cup();self.seek(self.frame)

    def change_cup(self, delta):
        self.select_cup((self.index + delta) % len(self.data.choices))

    def draw_overview(self):
        meta = json.loads((self.data.root / 'panorama_metadata.json').read_text())
        source = self.Image.open(self.data.root / 'panorama.png').convert('RGB')
        width, height = 285, 218
        scale = min(width / source.width, height / source.height)
        source = source.resize((round(source.width * scale), round(source.height * scale)), self.Image.Resampling.LANCZOS)
        self.overview_photo = self.ImageTk.PhotoImage(source)
        self.overview.delete('all');self.overview.create_image(0, 0, anchor='nw', image=self.overview_photo)
        transform = np.asarray(meta['raw_tracking_to_image_matrix'])
        centers = []
        for i, cup in enumerate(self.data.cups):
            raw = np.r_[np.asarray(cup['center_mm']) / self.data.mm_per_pixel, 1]
            point = transform @ raw;point = point[:2] / point[2] * scale
            radius = cup['radius_mm'] / self.data.mm_per_pixel * scale
            centers.append((point, radius))
            self.overview.create_oval(point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius,
                outline='#fc3344' if i == self.index else '#287c8e', width=3 if i == self.index else 1)
            self.overview.create_text(*point, text=str(cup['cup_id']), fill='#a00000' if i == self.index else '#17262c', font=('DejaVu Sans', 8, 'bold'))
        self.overview_centers = centers

    def overview_click(self, event):
        distance = [np.linalg.norm(point - [event.x, event.y]) / radius for point, radius in self.overview_centers]
        if min(distance) <= 1.2:
            self.select_cup(int(np.argmin(distance)))

    def request(self, frame):
        frame = int(np.clip(frame, 0, self.data.n - 1))
        camera = None if self.view.current() <= 0 else self.cameras[self.view.current() - 1]
        self.pending_token = self.worker.request(self.index, frame, camera, self.overlay_options())
        self.busy = True;self.request_started = time.monotonic()
        self.message.set(f'Loading {frame / self.data.fps / 3600:.6f} h…')

    def seek(self, frame):
        self.pause();self.request(frame)

    def step(self, direction):
        self.seek(self.frame + direction)

    def jump(self, seconds):
        self.seek(self.frame + round(seconds * self.data.fps))

    def goto_hour(self):
        try:
            self.seek(round(float(self.hour.get()) * 3600 * self.data.fps))
        except ValueError:
            self.message.set('Enter a numeric time in hours.')

    def pause(self):
        self.playing = False
        if self.play_after:
            self.root.after_cancel(self.play_after);self.play_after = None
        self.worker.generation += 1;self.pending_token = self.worker.generation;self.busy = False
        self.play_status.set('Paused')
        self.root.setvar('cup_video_playing', 0)

    def magnitude(self):
        return float(self.rate.get().replace('×', '').replace('x', ''))

    def play(self):
        self.direction = 1;self.begin_play()

    def begin_play(self):
        self.pause();self.playing = True
        self.root.setvar('cup_video_playing', 1)
        self.play_status.set(('Reverse ' if self.direction < 0 else 'Playing ') + self.rate.get())
        self.advance()

    def rewind(self):
        if self.playing and self.direction < 0:
            self.rate.set(f'{min(32, self.magnitude() * 2):g}×')
        self.direction = -1;self.begin_play()

    def fast_forward(self):
        self.rate.set(f'{min(32, max(2, self.magnitude() * 2)):g}×')
        self.direction = 1;self.begin_play()

    def toggle(self):
        self.pause() if self.playing else self.begin_play()

    def advance(self):
        self.play_after = None
        if self.closed or not self.playing or self.busy:
            return
        self.play_step = max(1, round(self.data.fps * self.magnitude() / 12))
        target = int(np.clip(self.frame + self.direction * self.play_step, 0, self.data.n - 1))
        if segment_at(self.data.segments, target)[0] is None:
            if self.direction > 0:
                next_start = [int(s['frame_offset']) for s in self.data.segments if int(s['frame_offset']) > target]
                if next_start:
                    target = min(next_start)
            else:
                previous_stop = [int(s['frame_offset']) + int(s['num_frames']) - 1 for s in self.data.segments
                                 if int(s['frame_offset']) + int(s['num_frames']) <= target]
                if previous_stop:
                    target = max(previous_stop)
        if target == self.frame:
            self.pause();return
        self.request(target)

    def poll(self):
        if self.closed:
            return
        while not self.worker.results.empty():
            token, index, frame, result, error = self.worker.results.get()
            if token != self.pending_token or index != self.index:
                continue
            self.busy = False
            if error:
                self.pause();self.image = None;self.canvas.delete('all')
                self.badge.configure(text='VIDEO ERROR', bg='#a64d4d');self.state_detail.set('No frame displayed')
                self.message.set(error);continue
            self.frame = frame;self.image, cameras, self.overlay_payload, self.rendered_overlay_options = result;self.render_count += 1
            from tracking.single_ant.cup_detections import draw_overlay, overlay_summary
            shown_camera = None if self.view.current() <= 0 else self.cameras[self.view.current()-1]
            masks = self.detections.ownership(index,shown_camera)
            display_options = dict(self.rendered_overlay_options)
            if shown_camera is not None:display_options['all_cameras'] = False
            self.overlay_picture = draw_overlay(self.image,self.overlay_payload,display_options,masks)
            self.overlay_detail.set(overlay_summary(self.overlay_payload) if any(self.rendered_overlay_options.get(k) for k in ['aruco','sleap','track','trail']) else 'Detection overlays off')
            category, text, detail = self.data.status(index, frame)
            self.badge.configure(text=text, bg='#3f9a79' if self.labels['video_motion'][frame] else COLORS[category]);self.state_detail.set(detail)
            segment, local = segment_at(self.data.segments, frame)
            source = self.data.segments[segment].get('block', 'recording') if segment is not None else 'UNRECORDED GAP'
            self.hour.set(f'{frame / self.data.fps / 3600:.6f}')
            timestamp = self.data.origin + timedelta(seconds=frame / self.data.fps)
            self.clock.set(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} JST   |   {source}   |   global frame {frame:,}"
                           + (f'   |   source frame {local:,}' if local is not None else ''))
            self.message.set(('Cup panorama · ' if self.view.current() == 0 else '')
                + ('cameras ' + ', '.join(f'{c:02d}' for c in cameras) if cameras else 'No video was recorded here')
                + '   |   Purple: S0/S1 · amber: not S0/S1 · gray: ambiguous   |   Click either timeline to seek')
            self.draw_video();self.draw_timelines()
            self.root.setvar('cup_video_aruco_count',len(self.overlay_payload['aruco']))
            self.root.setvar('cup_video_sleap_count',len(self.overlay_payload['poses']))
            self.root.setvar('cup_video_overlay_frame',self.overlay_payload['frame'])
            for key, value in [('cup_video_frame', frame), ('cup_video_cup', self.data.choices[index]['cup']),
                               ('cup_video_category', category), ('cup_video_render_count', self.render_count)]:
                self.root.setvar(key, value)
            if self.playing:
                elapsed = time.monotonic() - self.request_started
                interval = self.play_step / self.data.fps / self.magnitude()
                self.play_after = self.root.after(max(1, round(1000 * (interval - elapsed))), self.advance)
        self.root.after(20, self.poll)

    def label_change(self, direction):
        changes = self.labels['changes']
        if direction > 0:
            i = np.searchsorted(changes, self.frame, side='right')
            if i < len(changes):
                self.seek(changes[i])
        else:
            i = np.searchsorted(changes, self.frame, side='left') - 1
            if i >= 0:
                self.seek(changes[i])

    def sleep_jump(self, direction):
        starts = self.labels['sleep'][:, 0]
        i = np.searchsorted(starts, self.frame, side='right' if direction > 0 else 'left')
        if direction < 0:
            i -= 1
        if 0 <= i < len(starts):
            self.seek(starts[i])

    def set_span(self, event=None):
        self.local_span = int(self.span.get().split()[0]) * 60 * self.data.fps
        self.last_local_key = None;self.draw_timelines()

    def draw_timelines(self, force=False):
        if not hasattr(self, 'labels'):
            return
        width = max(10, self.full_timeline.winfo_width())
        if force or width != self.last_full_width:
            self.full_photo = self.ImageTk.PhotoImage(timeline_image(self.labels['category'], 0, self.data.n, width, 29))
            self.full_timeline.delete('all');self.full_timeline.create_image(0, 0, image=self.full_photo, anchor='nw')
            self.last_full_width = width
        self.full_timeline.delete('cursor')
        x = self.frame / max(1, self.data.n - 1) * width
        self.full_timeline.create_line(x, 0, x, 31, fill='#111', width=2, tags='cursor')
        # Recenter infrequently, keeping the local strip stable during playback.
        chunk = max(1, round(self.local_span / 5))
        key = int(self.frame // chunk), width, self.local_span
        if force or key != self.last_local_key:
            center = int(self.frame // chunk) * chunk
            lo = max(0, min(center - round(self.local_span / 2), self.data.n - round(self.local_span)))
            hi = min(self.data.n, lo + round(self.local_span))
            self.local_bounds = lo, hi
            self.local_photo = self.ImageTk.PhotoImage(timeline_image(self.labels['category'], lo, hi, width, 37))
            self.local_timeline.delete('all');self.local_timeline.create_image(0, 0, image=self.local_photo, anchor='nw')
            self.last_local_key = key
        lo, hi = self.local_bounds
        self.local_timeline.delete('cursor');x = (self.frame - lo) / max(1, hi - lo) * width
        self.local_timeline.create_line(x, 0, x, 39, fill='#111', width=2, tags='cursor')
        self.timeline_text.set(f'Whole recording: 0–{self.data.n / self.data.fps / 3600:.2f} h   |   Local view: {lo / self.data.fps / 3600:.4f}–{hi / self.data.fps / 3600:.4f} h')

    def fit(self):
        self.zoom = 1.;self.focus[:] = .5;self.draw_video()

    def zoom_by(self, factor, event=None):
        if event is not None and self.draw_box:
            left, top, side, x0, y0, span = self.draw_box
            self.focus = np.clip([(x0 + (event.x - left) / side * span) / self.data.size,
                                  (y0 + (event.y - top) / side * span) / self.data.size], 0, 1)
        self.zoom = float(np.clip(self.zoom * factor, 1, 8));self.draw_video()

    def pan_start(self, event):
        self.canvas.focus_set();self.drag_origin = event.x, event.y, self.focus.copy()

    def pan_drag(self, event):
        if not self.drag_origin or not self.draw_box:
            return
        x, y, focus = self.drag_origin
        side = self.draw_box[2]
        self.focus = np.clip(focus - np.array([event.x - x, event.y - y]) / side / self.zoom, 0, 1)
        self.draw_video()

    def draw_video(self):
        if self.image is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        side = max(1, min(width, height) - 8);size = self.data.size;span = size / self.zoom
        origin = np.clip(self.focus * size - span / 2, 0, size - span)
        source = self.overlay_picture if self.overlay_picture is not None else self.Image.fromarray(self.image)
        picture = source.crop((origin[0], origin[1], origin[0] + span, origin[1] + span))
        picture = picture.resize((side, side), self.Image.Resampling.BILINEAR)
        self.photo = self.ImageTk.PhotoImage(picture)
        self.canvas.delete('all');left, top = (width - side) / 2, (height - side) / 2
        self.canvas.create_image(left, top, anchor='nw', image=self.photo)
        self.draw_box = left, top, side, origin[0], origin[1], span

    def close(self):
        self.pause();self.closed = True;self.worker.close();self.root.destroy()


def main():
    import argparse
    import tkinter as tk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--cup', type=int, default=30)
    parser.add_argument('--start-hours', type=float, default=0)
    parser.add_argument('--cup-motion', action='store_true', help='Overlay validated video-motion exclusions of original ambiguous time')
    parser.add_argument('--size', type=int, default=1000, help='Cup panorama resolution in pixels')
    args = parser.parse_args()
    root = tk.Tk()
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    CupVideoViewer(root, args.dataset, args.cup, round(args.start_hours * 3600 * float(manifest['fps'])), args.size, args.cup_motion)
    root.mainloop()


if __name__ == '__main__':
    main()
