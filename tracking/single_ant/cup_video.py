"""Frame-accurate local cup video decoding, including unindexed recorder MKVs.

The recorder files have no usable duration/cue index. Seek by Matroska cluster
timestamps, start on a real keyframe, and verify every decoded presentation
timestamp against the hardware-trigger frame number. Never use OpenCV seeks.
"""
from __future__ import annotations
from collections import OrderedDict
import io
from pathlib import Path

import numpy as np


class Cancelled(Exception):
    pass


def vint_bytes(data, offset=0, identifier=False):
    if offset >= len(data) or data[offset] == 0:
        raise ValueError('Invalid/truncated EBML integer')
    width = 1
    while not data[offset] & (1 << (8 - width)):
        width += 1
    if offset + width > len(data):
        raise ValueError('Truncated EBML integer')
    value = int.from_bytes(data[offset:offset + width], 'big')
    return (value if identifier else value & ((1 << (7 * width)) - 1)), width


def element(file):
    start = file.tell()
    first = file.read(1)
    if not first:
        raise EOFError
    if first == b'\0':
        raise ValueError(f'Invalid Matroska element at byte {start}')
    width = 1
    while not first[0] & (1 << (8 - width)):
        width += 1
    identifier = int.from_bytes(first + file.read(width - 1), 'big')
    first = file.read(1)
    if not first or first == b'\0':
        raise ValueError('Missing EBML element size')
    width = 1
    while not first[0] & (1 << (8 - width)):
        width += 1
    size = int.from_bytes(first + file.read(width - 1), 'big') & ((1 << (7 * width)) - 1)
    return start, identifier, size, file.tell()


def cluster_description(data, absolute, file_size):
    """Validate a candidate cluster and read its first video block timestamp."""
    identifier, w = vint_bytes(data, 0, True)
    size, v = vint_bytes(data, w)
    if identifier != 0x1F43B675 or size > 256 * 1024 * 1024 or absolute + w + v + size > file_size:
        raise ValueError('Not a complete recorder cluster')
    offset, timestamp = w + v, None
    for _ in range(8):
        child, cw = vint_bytes(data, offset, True)
        count, sw = vint_bytes(data, offset + cw)
        begin = offset + cw + sw
        if child == 0xE7:
            if not 1 <= count <= 8 or begin + count > len(data):
                raise ValueError('Bad cluster timestamp')
            timestamp = int.from_bytes(data[begin:begin + count], 'big')
        if child == 0xA3:
            track, tw = vint_bytes(data, begin)
            if timestamp is None or track != 1 or begin + tw + 3 > len(data):
                raise ValueError('Unsupported video block')
            relative = int.from_bytes(data[begin + tw:begin + tw + 2], 'big', signed=True)
            flags = data[begin + tw + 2]
            return dict(offset=absolute, end=absolute + w + v + size,
                        milliseconds=timestamp + relative, key=bool(flags & 0x80))
        if count > 64 or begin + count > len(data):
            raise ValueError('Unsupported cluster prefix')
        offset = begin + count
    raise ValueError('Cluster has no initial SimpleBlock')


class MatroskaSource:
    MAGIC = b'\x1f\x43\xb6\x75'

    def __init__(self, path, fps, n_frames):
        self.path, self.fps, self.n = Path(path), float(fps), int(n_frames)
        self.size = self.path.stat().st_size
        self.keys = {0: None}
        with self.path.open('rb') as file:
            start, kind, size, payload = element(file)
            if kind != 0x1A45DFA3:
                raise ValueError(f'Expected a Matroska video: {path}')
            file.seek(0);header = file.read(payload + size)
            _, kind, _, _ = element(file)
            if kind != 0x18538067:
                raise ValueError('Missing Matroska segment')
            extras = []
            while file.tell() < self.size:
                start, kind, size, payload = element(file)
                if kind == 0x1F43B675:
                    self.first = start
                    self.keys[0] = start
                    break
                if kind in [0x1549A966, 0x1654AE6B]:
                    file.seek(start);extras.append(file.read(payload + size - start))
                file.seek(payload + size)
            else:
                raise ValueError('Video has no Matroska clusters')
        # A clean streaming header deliberately omits invalid source seek heads.
        self.prefix = header + b'\x18\x53\x80\x67\x01\xff\xff\xff\xff\xff\xff\xff' + b''.join(extras)

    def after(self, file, position):
        file.seek(max(self.first, int(position)))
        carry = b''
        for _ in range(20):
            begin = file.tell() - len(carry)
            data = carry + file.read(262144)
            if len(data) <= len(carry):
                raise EOFError
            offset = data.find(self.MAGIC)
            while offset >= 0:
                try:
                    found = cluster_description(data[offset:offset + 512], begin + offset, self.size)
                    if 0 <= found['milliseconds'] <= self.n / self.fps * 1000 + 2000:
                        return found
                except ValueError:
                    pass
                offset = data.find(self.MAGIC, offset + 1)
            carry = data[-512:]
        raise ValueError('Cannot locate a Matroska cluster near seek position')

    def near_time(self, file, milliseconds, cancelled=lambda: False):
        if milliseconds <= 0:
            return self.after(file, self.first)
        lo, hi, lt, ht = self.first, self.size, 0., self.n / self.fps * 1000
        best = self.after(file, self.first)
        for _ in range(30):
            if cancelled():
                raise Cancelled
            if hi - lo < 262144:
                break
            fraction = np.clip((milliseconds - lt) / max(1, ht - lt), .05, .95)
            guess = int(lo + fraction * (hi - lo))
            try:
                point = self.after(file, guess)
            except EOFError:
                hi = guess
                continue
            time = point['milliseconds']
            if time <= milliseconds:
                best = point;lo = point['end'];lt = time
            else:
                hi = point['offset'];ht = time
            if abs(time - milliseconds) < 150:
                best = point
                break
        return best

    def keyframe(self, frame, cancelled=lambda: False):
        """Locate a preceding real keyframe with a bounded, buffered scan."""
        target = frame / self.fps * 1000
        lookback = 65000.
        with self.path.open('rb') as file:
            while True:
                point = self.near_time(file, max(0, target - lookback), cancelled)
                best = None
                file.seek(point['offset'])
                carry = b''
                finished = False
                while file.tell() < self.size:
                    if cancelled():
                        raise Cancelled
                    begin = file.tell() - len(carry)
                    data = carry + file.read(4 * 1024 * 1024)
                    offset = data.find(self.MAGIC)
                    while offset >= 0:
                        try:
                            point = cluster_description(data[offset:offset + 512], begin + offset, self.size)
                            if point['milliseconds'] > target + 1:
                                finished = True
                                break
                            if point['key']:
                                best = point
                        except ValueError:
                            pass
                        offset = data.find(self.MAGIC, offset + 1)
                    if finished:
                        break
                    carry = data[-512:]
                if best is not None:
                    return best
                if lookback >= target:
                    raise ValueError('No keyframe found before requested trigger frame')
                lookback *= 2


class ClusterStream(io.RawIOBase):
    def __init__(self, source, offset):
        super().__init__()
        self.prefix = io.BytesIO(source.prefix)
        self.file = source.path.open('rb');self.file.seek(offset)

    def readable(self):
        return True

    def seekable(self):
        return False

    def read(self, size=-1):
        data = self.prefix.read(size)
        if size < 0:
            return data + self.file.read()
        return data + self.file.read(size - len(data))

    def close(self):
        if not self.closed:
            self.file.close();self.prefix.close()
        super().close()


class CameraDecoder:
    def __init__(self, path, fps, n_frames, map_x, map_y, cache_frames=96):
        import av
        self.av = av
        self.source = MatroskaSource(path, fps, n_frames)
        self.map_x, self.map_y = map_x, map_y
        self.limit = cache_frames;self.cache = OrderedDict()
        self.container = self.stream = self.iterator = self.input = None
        self.last = -1

    def close(self):
        if self.container is not None:
            self.container.close()
        if self.input is not None:
            self.input.close()
        self.container = self.iterator = self.input = None;self.last = -1

    def restart(self, frame, cancelled):
        self.close()
        key = self.source.keyframe(frame, cancelled)
        self.input = ClusterStream(self.source, key['offset'])
        self.container = self.av.open(self.input, format='matroska', options={'probesize': '32768', 'analyzeduration': '0'})
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 4
        self.stream.codec_context.thread_type = 'AUTO'
        self.iterator = iter(self.container.decode(self.stream))
        self.last = int(round(key['milliseconds'] * self.source.fps / 1000)) - 1

    def read(self, frame, cancelled=lambda: False):
        import cv2
        frame = int(frame)
        if frame in self.cache:
            self.cache.move_to_end(frame)
            return self.cache[frame]
        if self.container is None or frame <= self.last or frame - self.last > self.source.fps * 3:
            self.restart(frame, cancelled)
        context = self.stream.codec_context
        context.skip_frame = 'NONREF' if frame - self.last > 72 else 'DEFAULT'
        for decoded in self.iterator:
            if cancelled():
                self.close()
                raise Cancelled
            if decoded.pts is None:
                raise ValueError('Video has a frame without a presentation timestamp')
            trigger_position = float(decoded.pts * decoded.time_base) * self.source.fps
            number = int(round(trigger_position))
            if abs(trigger_position - number) > .05:
                self.close()
                raise ValueError('Video timestamps are not on the tracking trigger-frame grid')
            self.last = number
            if number >= frame - 72:
                context.skip_frame = 'DEFAULT'
                gray = decoded.to_ndarray(format='gray')
                crop = cv2.remap(gray, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.cache[number] = crop
                while len(self.cache) > self.limit:
                    self.cache.popitem(last=False)
            if number == frame:
                return self.cache[number]
            if number > frame:
                self.close()
                raise ValueError(f'Decoder skipped trigger frame {frame}; first returned {number}')
        raise ValueError(f'Video ended before trigger frame {frame}')
