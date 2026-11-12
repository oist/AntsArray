#!/usr/bin/env python3
"""Inspect cached single-ant speeds: python speed_viewer.py RESULTS_FOLDER."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import numpy as np
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tracking.single_ant.state_overlay import load_s0, load_sleep_categories, clip_intervals, state_status


def speed_window(values, fps, start_h, stop_h, segments, max_bins=2000):
    """All samples contribute; retain extremes, missing bins and recording gaps."""
    if fps <= 0 or max_bins <= 0:
        raise ValueError('FPS and display bin count must be positive')
    first = max(0, int(np.floor(start_h*3600*fps+1e-6)))
    last = min(len(values), int(np.ceil(stop_h*3600*fps-1e-6)))
    width = max(1, int(np.ceil(max(0, last-first)/max_bins)))
    parts = []
    for segment in segments:
        lo = max(first, segment['frame_offset'])
        hi = min(last, segment['frame_offset']+segment['num_frames'])
        if lo >= hi:
            continue
        v = np.asarray(values[lo:hi])
        valid = np.isfinite(v)
        starts = np.arange(0, len(v), width)
        lengths = np.minimum(width, len(v)-starts)
        counts = np.add.reduceat(valid.astype(np.int64), starts)
        total = np.add.reduceat(np.where(valid, v, 0), starts, dtype=np.float64)
        mean = np.divide(total, counts, out=np.full(len(starts), np.nan), where=counts>0)
        low = np.minimum.reduceat(np.where(valid, v, np.inf), starts).astype(float)
        high = np.maximum.reduceat(np.where(valid, v, -np.inf), starts).astype(float)
        low[counts == 0] = np.nan; high[counts == 0] = np.nan
        x = (lo+starts+(lengths-1)/2)/fps/3600
        if parts:
            # An explicit NaN prevents a line spanning even a gap shorter than
            # an overview bin, including boundaries between adjacent blocks.
            parts.append(np.full((1, 5), np.nan))
        parts.append(np.column_stack([x, mean, low, high, 100*counts/lengths]))
    return (np.concatenate(parts) if parts else np.empty((0, 5))), width


class SpeedData:
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        self.root = path.parent if path.is_file() else path
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.fps, self.n = float(self.manifest['fps']), int(self.manifest['n_frames'])
        if self.fps <= 0 or self.n <= 0:
            raise ValueError('Results need a positive frame count and frame rate')
        self.duration = self.n/self.fps/3600
        self.origin = datetime.fromisoformat(self.manifest['start_datetime'])
        self.segments = self.manifest.get('recording_segments') or [dict(frame_offset=0, num_frames=self.n)]
        with (self.root/'cup_identities.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        cups = self.manifest.get('physical_cups', self.manifest['cups'])
        self.choices = []
        for cid in sorted({c['cup_id'] for c in cups}):
            found = [r for r in rows if int(r['CupID']) == cid and r['Eligible'].lower() == 'true']
            for row in found:
                tid = int(row.get('TrackID') or cid)
                tag = int(row['ArucoID'])
                self.choices.append(dict(cup=cid, track=tid, label=f'Cup {cid} — ID {tag}'))
            if not found:
                self.choices.append(dict(cup=cid, track=None, label=f'Cup {cid} — unassigned'))
        if not self.choices:
            raise ValueError('No cup identities found in these results')
        self.loaded_key, self.loaded = None, None
        self.state_key, self.state_data = None, None

    def speed(self, index, unclipped=False):
        tid = self.choices[index]['track']
        if tid is None:
            return None
        key = tid, unclipped
        if key != self.loaded_key:
            filename = 'speed_unclipped_mm_s.npy' if unclipped else 'speed_mm_s.npy'
            # Use the selected local folder, regardless of metadata mount paths.
            path = self.root/'analysis/speed_vectors/per_track'/f'TrackID_{tid:04d}_all_single'/filename
            values = np.load(path, mmap_mode='r')
            if values.shape != (self.n,):
                raise ValueError(f'Speed vector has the wrong recording span: {path}')
            self.loaded, self.loaded_key = values, key
        return self.loaded

    def states(self, index):
        tid = self.choices[index]['track']
        if self.state_key != tid or self.state_data is None:
            self.state_data = load_sleep_categories(self.root,tid)
            if self.state_data is None:self.state_data=load_s0(self.root,tid)
            self.state_key = tid
            if self.state_data is not None and self.state_data['fps'] != self.fps:
                raise ValueError('State labels and speed have different frame rates')
        return self.state_data

    def s0(self,index):
        return load_s0(self.root,self.choices[index]['track'])


class SpeedViewer:
    def __init__(self, root, path, cup=None):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.root, self.pending, self.fit_pending = root, None, False
        root.title('Single-ant speed inspection')
        root.geometry('1500x950+180+100')
        bar = ttk.Frame(root, padding=8); bar.pack(fill='x')
        ttk.Button(bar, text='Open…', command=self.open).pack(side='left')
        ttk.Button(bar, text='Previous cup', command=lambda: self.step(-1)).pack(side='left', padx=(10, 2))
        self.selector = ttk.Combobox(bar, state='readonly', width=26)
        self.selector.pack(side='left'); self.selector.bind('<<ComboboxSelected>>', lambda e: self.schedule(True))
        ttk.Button(bar, text='Next cup', command=lambda: self.step(1)).pack(side='left', padx=2)
        self.unclipped, self.log = tk.BooleanVar(), tk.BooleanVar()
        self.show_s0 = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text='Unclipped speed', variable=self.unclipped, command=lambda: self.schedule(True)).pack(side='left', padx=12)
        ttk.Checkbutton(bar, text='Log scale', variable=self.log, command=lambda: self.schedule(True)).pack(side='left')
        ttk.Checkbutton(bar, text='Sleep states', variable=self.show_s0, command=self.schedule).pack(side='left',padx=12)
        ranges = ttk.Frame(root, padding=(8, 0, 8, 8)); ranges.pack(fill='x')
        self.start, self.stop = tk.StringVar(value='0'), tk.StringVar()
        for label, variable in [('Start (hours)', self.start), ('End (hours)', self.stop)]:
            ttk.Label(ranges, text=label).pack(side='left', padx=(8, 4))
            entry = ttk.Entry(ranges, textvariable=variable, width=12); entry.pack(side='left')
            entry.bind('<Return>', self.apply_window)
        ttk.Button(ranges, text='Go', command=self.apply_window).pack(side='left', padx=8)
        ttk.Button(ranges, text='Full recording', command=self.full).pack(side='left')
        ttk.Button(ranges, text='Fit Y', command=lambda: self.schedule(True)).pack(side='left', padx=8)
        self.figure = Figure(figsize=(14, 8), layout='constrained')
        self.ax, self.coverage = self.figure.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[4, 1]))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        # Tk can change display DPI on Map without another Configure event.
        # Fit the renderer to the actual widget after that change settles.
        self.canvas.get_tk_widget().bind('<Map>', lambda event: root.after_idle(self.sync_canvas),add='+')
        self.status = tk.StringVar()
        self.status_label = ttk.Label(root, textvariable=self.status, padding=8)
        self.status_label.bind('<Configure>',lambda e:self.status_label.configure(wraplength=max(200,e.width-20)))
        self.status_label.pack(side='bottom', fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.toolbar.update(); self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.load(path, cup)

    def sync_canvas(self):
        from types import SimpleNamespace
        widget=self.canvas.get_tk_widget()
        width,height=widget.winfo_width(),widget.winfo_height()
        if width>1 and height>1:
            self.canvas.resize(SimpleNamespace(width=width,height=height))

    def load(self, path, cup=None):
        self.data = SpeedData(path)
        self.root.setvar('single_ant_dataset',str(self.data.root))
        self.selector['values'] = [c['label'] for c in self.data.choices]
        self.selector.current(next((i for i, c in enumerate(self.data.choices) if c['cup'] == cup), 0))
        self.ax.clear(); self.coverage.clear()
        self.line, = self.ax.plot([], [], color='#19745c', lw=1, label='mean')
        self.band = None
        self.state_artists = []
        self.cover, = self.coverage.plot([], [], color='#426d9c', lw=.9)
        self.message = self.ax.text(.5, .5, '', transform=self.ax.transAxes, ha='center', va='center')
        self.ax.set(ylabel='Speed (mm/s)', ylim=(0, 20))
        self.ax.set_autoscalex_on(False)
        self.coverage.set(ylabel='Valid speed (%)', ylim=(-2, 102),
                          xlabel='Hours since '+self.data.origin.strftime('%Y-%m-%d %H:%M:%S %z'))
        previous = 0
        for s in self.data.segments:
            start = s['frame_offset']
            for ax in [self.ax, self.coverage]:
                if start > previous:
                    ax.axvspan(previous/self.data.fps/3600, start/self.data.fps/3600, color='grey', alpha=.22)
                if start:
                    ax.axvline(start/self.data.fps/3600, color='grey', lw=.6, ls='--')
            previous = start+s['num_frames']
        self.ax.format_coord = self.cursor
        self.ax.set_xlim(0, self.data.duration)
        for ax in [self.ax, self.coverage]:
            ax.callbacks.connect('xlim_changed', lambda ax: self.schedule())
        self.toolbar.update()
        self.full()

    def cursor(self, x, y):
        clock = self.data.origin+timedelta(hours=float(x))
        frame = int(np.floor(x*3600*self.data.fps))
        state = (self.data.state_data if self.data.state_key==self.data.choices[self.selector.current()]['track'] else None)
        label=state_status(state,frame)
        return f'{clock:%Y-%m-%d %H:%M:%S} | {x:.6f} h | frame {frame} | {y:.3f} mm/s | {label}'

    def overlay_s0(self, labels, lo, hi):
        from matplotlib.collections import PolyCollection
        for artist in self.state_artists:artist.remove()
        self.state_artists = []
        if not self.show_s0.get():return
        def rectangles(spans, bottom, top, color, alpha, label=None):
            spans = np.asarray(spans,float).reshape(-1,2)/self.data.fps/3600
            vertices = np.empty((len(spans),4,2))
            vertices[:,:,0]=spans[:,[0,1,1,0]]
            vertices[:,:,1]=[bottom,bottom,top,top]
            collection=PolyCollection(vertices,facecolors=color,edgecolors='none',alpha=alpha,
                transform=self.ax.get_xaxis_transform(),zorder=.2,label=label)
            self.ax.add_collection(collection,autolim=False);self.state_artists.append(collection)
        first,stop=int(np.floor(lo*3600*self.data.fps)),int(np.ceil(hi*3600*self.data.fps))
        rectangles([[first,stop]],.975,1,'#d9dde3',1)
        if labels is not None:
            known=clip_intervals(labels['known'],first,stop)
            if labels.get('mode')=='sleep01':
                sleep=clip_intervals(labels['sleep'],first,stop);not_sleep=clip_intervals(labels['not_sleep'],first,stop)
                rectangles([[first,stop]],0,.975,'#d9dde3',.1,'Ambiguous')
                rectangles(not_sleep,.975,1,'#d89435',1)
                rectangles(sleep,.975,1,'#9162c6',1)
                rectangles(not_sleep,0,.975,'#d89435',.09,'Not S0/S1')
                rectangles(sleep,0,.975,'#9162c6',.18,'S0/S1 (sleep)')
            else:
                s0=clip_intervals(labels['s0'],first,stop)
                rectangles(known,.975,1,'white',1)
                rectangles(s0,.975,1,'#9162c6',1)
                rectangles(s0,0,.975,'#9162c6',.18,'S0')

    def step(self, delta):
        self.selector.current((self.selector.current()+delta) % len(self.data.choices))
        self.schedule(True)

    def open(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askdirectory(initialdir=self.data.root, title='Select single-ant results folder')
        if path:
            try:
                self.load(path)
            except (OSError, ValueError, KeyError) as error:
                messagebox.showerror('Cannot open results', str(error), parent=self.root)

    def apply_window(self, event=None):
        from tkinter import messagebox
        try:
            lo, hi = float(self.start.get()), float(self.stop.get())
            if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi <= self.data.duration+1e-5:
                raise ValueError(f'Choose 0 ≤ start < end ≤ {self.data.duration:.6f} hours.')
            self.ax.set_xlim(lo, min(hi, self.data.duration))
        except ValueError as error:
            messagebox.showerror('Time range', str(error), parent=self.root)

    def full(self):
        self.ax.set_xlim(0, self.data.duration)
        self.schedule(True)

    def schedule(self, fit=False):
        self.fit_pending |= fit
        if self.pending:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(100, self.redraw)

    def redraw(self):
        from tkinter import messagebox
        self.pending = None
        fit, self.fit_pending = self.fit_pending, False
        choice = self.data.choices[self.selector.current()]
        try:
            values = self.data.speed(self.selector.current(), self.unclipped.get())
            labels = self.data.states(self.selector.current()) if self.show_s0.get() else None
            lo, hi = self.ax.get_xlim()
            points, width = (speed_window(values, self.data.fps, lo, hi, self.data.segments)
                             if values is not None else (np.empty((0, 5)), 1))
        except (OSError, ValueError) as error:
            messagebox.showerror('Cannot load speed', str(error), parent=self.root)
            return
        self.line.set_data(points[:, 0], points[:, 1])
        if self.band is not None:
            self.band.remove()
        self.band = self.ax.fill_between(points[:, 0], points[:, 2], points[:, 3], color='#19745c', alpha=.22, label='min–max')
        self.cover.set_data(points[:, 0], points[:, 4])
        self.overlay_s0(labels,lo,hi)
        finite = points[:, 3][np.isfinite(points[:, 3])]
        self.message.set_text('No qualified ArUco ID for this cup' if values is None else 'No valid speeds in this window')
        self.message.set_visible(not len(finite))
        if self.log.get():
            self.ax.set_yscale('symlog', linthresh=1)
        else:
            self.ax.set_yscale('linear')
        if fit:
            upper = max(1., float(finite.max())*1.05) if self.unclipped.get() and len(finite) else 20.
            self.ax.set_ylim(0, upper)
        self.start.set(f'{lo:.6f}'); self.stop.set(f'{hi:.6f}')
        mode = 'Unclipped' if self.unclipped.get() else 'Standard colony'
        self.figure.suptitle(f"{self.data.manifest.get('title', 'Single ants')} — {choice['label']} — {mode.lower()} speed")
        self.ax.legend(loc='upper right')
        resolution = 'individual frames' if width == 1 else f'bins of {width/self.data.fps:.3g} seconds'
        state_note = (' | purple = S0; top strip grey = state unknown' if self.show_s0.get() else '')
        if self.show_s0.get() and labels is not None and labels.get('mode')=='sleep01':
            counts=np.zeros(3,np.int64);inferred=0
            for segment in self.data.segments:
                a=max(0,int(np.floor(lo*3600*self.data.fps)),segment['frame_offset'])
                b=min(self.data.n,int(np.ceil(hi*3600*self.data.fps)),segment['frame_offset']+segment['num_frames'])
                if b>a:
                    counts+=np.bincount(labels['category'][a:b]+1,minlength=3)
                    inferred+=np.count_nonzero(labels['evidence'][a:b]&6)
            total=max(1,counts.sum());limits=labels['thresholds']
            state_note=(f' | S0/S1 sleep {counts[2]/total:.1%} · not S0/S1 {counts[1]/total:.1%} · ambiguous {counts[0]/total:.1%}'
                f' | motion-filled not {inferred/total:.1%} | fast: velocity >{limits["velocity"]:.2f}, antenna >{limits["antenna"]:.2f} mm/s'
                ' | purple=sleep · amber=not · grey=ambiguous; hover for evidence')
        if self.show_s0.get() and (labels is None or not len(labels['known'])):
            state_note+=' | no state assignments for this cup'
        self.status.set(f'{mode} speed | {resolution}: mean and min–max range | grey background = not recorded'+state_note+' | toolbar pan/zoom')
        self.canvas.draw_idle()


def main():
    import tkinter as tk
    from tkinter import filedialog
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='?')
    parser.add_argument('--cup', type=int)
    args = parser.parse_args()
    root = tk.Tk()
    path = args.results
    if path is None:
        root.withdraw()
        path = filedialog.askdirectory(title='Select single-ant results folder')
        if not path:
            root.destroy(); return
        root.deiconify()
    SpeedViewer(root, path, args.cup)
    root.mainloop()


if __name__ == '__main__':
    main()
