"""Frame-exact raw detections and tracking overlays for the cup video viewer."""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

DEFAULT_NAMES = ['aruco', 'occiput', 'petiole', 'gaster_tip', 'antenna_L_1',
                 'antenna_L_2', 'antenna_L_3', 'antenna_R_1', 'antenna_R_2', 'antenna_R_3']
# These SLEAP models declare node names but no edges. Anatomical display chains.
DEFAULT_EDGES = [(1, 0), (0, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 7), (7, 8), (8, 9)]
RAW_TAG = '#6aee58'
RAW_POSE = '#f267d5'
TRACK_POSE = '#38dded'
TRACK_POSITION = '#ffdf47'


def project_points(xy, matrix):
    xy = np.asarray(xy, float).reshape(-1, 2)
    homogeneous = np.c_[xy, np.ones(len(xy))] @ np.asarray(matrix).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return homogeneous[:, :2] / homogeneous[:, 2:3]


def cup_coordinates(mm, cup, size):
    radius = float(cup['radius_mm']) * 1.06
    return (np.asarray(mm) - np.asarray(cup['center_mm'])) / (2 * radius) * size + (size - 1) / 2


def near_cup(mm, cup):
    return np.isfinite(mm).all(axis=-1) & (np.linalg.norm(np.asarray(mm) - cup['center_mm'], axis=-1) <= cup['radius_mm'] * 1.06)


class FrameTable:
    """Index the exact exported compound dataset, including duplicate tag IDs."""
    def __init__(self, path, dataset, expected_frames, frame_attribute):
        import h5py
        self.file = h5py.File(path, 'r')
        try:
            if int(self.file.attrs[frame_attribute]) != expected_frames:
                raise ValueError(f'Detection frame span changed: {path}')
            self.table = self.file[dataset]
            self.frames = self.table.fields('Frame')[:]
            if len(self.frames) and (self.frames.min() < 0 or self.frames.max() >= expected_frames):
                raise ValueError(f'Detection frame outside its source chunk: {path}')
            self.order = None
            if (np.diff(self.frames.astype(np.int64)) < 0).any():
                self.order = np.argsort(self.frames, kind='stable')
                self.frames = self.frames[self.order]
        except BaseException:
            self.file.close();raise

    def frame(self, number):
        lo, hi = np.searchsorted(self.frames, [number, number + 1])
        if self.order is None:return self.table[lo:hi]
        return self.table[np.sort(self.order[lo:hi])]

    def close(self):self.file.close()


class TrackTable:
    """Read just the row groups intersecting the frame/trail; no whole-track scan."""
    def __init__(self, path):
        import pyarrow.parquet as pq
        self.file = pq.ParquetFile(path)
        self.cache = OrderedDict()
        column = self.file.schema_arrow.get_field_index('Frame')
        self.ranges = []
        for i in range(self.file.num_row_groups):
            stats = self.file.metadata.row_group(i).column(column).statistics
            if stats is None or not stats.has_min_max:
                raise ValueError('Track parquet has no frame index')
            self.ranges.append((int(stats.min), int(stats.max)))

    def frame(self, number, trail_frames=0):
        columns = ['Frame','Bodypoint','X','Y','TrackX','TrackY','ArucoX','ArucoY',
                   'CameraID','ArucoCam','SleapCam','PositionSource','TrackID','ArucoID']
        columns = [x for x in columns if x in self.file.schema_arrow.names]
        parts = []
        for i, (lo, hi) in enumerate(self.ranges):
            if hi < number - trail_frames or lo > number:continue
            if i not in self.cache:
                self.cache[i] = self.file.read_row_group(i, columns=columns).to_pandas()
            self.cache.move_to_end(i)
            while len(self.cache) > 3:self.cache.popitem(last=False)
            data = self.cache[i]
            parts.append(data.loc[data.Frame.between(number - trail_frames, number)])
        if not parts:return None, np.empty((0,3))
        import pandas as pd
        rows = pd.concat(parts, ignore_index=True)
        current = rows.loc[rows.Frame.eq(number)]
        trail = rows.drop_duplicates('Frame')[['Frame','TrackX','TrackY']].to_numpy(float)
        return current, trail

    def close(self):self.file.close()


class DetectionReader:
    """All I/O runs in the video worker, aligned through the tracking manifests."""
    def __init__(self, data):
        self.data = data;self.manifests = {};self.tables = OrderedDict();self.tracks = OrderedDict()
        self.names = DEFAULT_NAMES;self.edges = DEFAULT_EDGES;self.geometry = {}

    def resolve(self, path):
        from tracking.single_ant.video_viewer import mounted_path
        return mounted_path(path, self.data.root)

    def manifest(self, segment):
        if segment not in self.manifests:
            block = self.resolve(self.data.segments[segment]['source_block'])
            contract_path = block / 'analysis_outputs/single_ant/manifest.json'
            # The stitched source registry is authoritative if the block was
            # published into a differently named analysis folder.
            for source in self.data.manifest.get('source_manifests', []):
                candidate = self.resolve(source['path'])
                contract = json.loads(candidate.read_text())
                if str(contract.get('source_block','')).split('/bucket/')[-1] == str(block).split('/bucket/')[-1]:
                    contract_path = candidate;break
            contract = json.loads(contract_path.read_text())
            path = contract_path.with_name('detections_manifest.json')
            if not path.exists():path = self.resolve(contract['detections'])
            info = json.loads(path.read_text())
            if (float(info['fps']) != self.data.fps
                    or int(info['n_frames']) != int(self.data.segments[segment]['num_frames'])
                    or not np.isclose(1 / float(info['board_px_per_mm']), self.data.mm_per_pixel)
                    or not np.allclose(info['homographies'], self.data.matrices)):
                raise ValueError('Detection timing/calibration differs from the displayed video')
            files = {}
            for item in info['files']:files.setdefault(int(item['camera_id']), []).append(item)
            for camera, items in files.items():
                items.sort(key=lambda x:x['frame_start']);end=0
                for item in items:
                    if item['frame_start'] != end:raise ValueError(f'Missing detection chunk for camera {camera+1}')
                    end += int(item['frames'])
                if end != info['n_frames']:raise ValueError('Incomplete detection source camera')
            first = next(iter(files.values()))[0]
            export = self.resolve(first['sleap'])
            slp = export.with_name(export.name.replace('_sleap_data.h5','.slp'))
            if slp.exists():
                import h5py
                with h5py.File(slp,'r') as f:
                    meta = json.loads(f['metadata'].attrs['json'])
                self.names = [node['name'].strip() for node in meta['nodes']]
                links = meta['skeletons'][0].get('links',[])
                if links:self.edges = [(int(e['source']),int(e['target'])) for e in links]
                else:
                    normalize = lambda name: ''.join(c.lower() for c in name if c.isalnum())
                    lookup = {normalize(name):i for i,name in enumerate(self.names)}
                    self.edges = [(lookup[normalize(DEFAULT_NAMES[a])],lookup[normalize(DEFAULT_NAMES[b])])
                        for a,b in DEFAULT_EDGES if normalize(DEFAULT_NAMES[a]) in lookup and normalize(DEFAULT_NAMES[b]) in lookup]
            self.manifests[segment] = dict(files=files, scale=float(contract['output_mm_per_px']), names=self.names, edges=self.edges)
        return self.manifests[segment]

    def table(self, item, kind):
        path = self.resolve(item[kind]);key = str(path)
        if key not in self.tables:
            dataset, attr = ('aruco_detections','num_frames') if kind=='aruco' else ('sleap_data','frame_count')
            self.tables[key] = FrameTable(path,dataset,int(item['frames']),attr)
        self.tables.move_to_end(key)
        while len(self.tables)>16:self.tables.popitem(last=False)[1].close()
        return self.tables[key]

    def frame(self, index, frame, camera, options, cancelled=lambda:False):
        from tracking.single_ant.video_viewer import segment_at
        output = dict(aruco=[], poses=[], track=None, trail=np.empty((0,3)), errors=[], frame=int(frame),
                      names=self.names, edges=self.edges, requested=options.copy(), gap=False)
        segment, local = segment_at(self.data.segments,frame)
        output['gap'] = segment is None
        if segment is None or not any(options.get(k) for k in ['aruco','sleap','track','trail']):return output
        cup = self.data.cups[index]
        try:source=self.manifest(segment)
        except Exception as error:
            output['errors'].append(str(error));return output
        output.update(names=source['names'],edges=source['edges'])
        plans, needed, _ = self.data.geometry(index)
        cameras = ([camera] if camera is not None else [p['camera'] for p in plans] if options.get('all_cameras') else needed)
        for cam in cameras:
            if cancelled():return output
            try:
                item=next((f for f in source['files'][cam] if f['frame_start']<=local<f['frame_start']+f['frames']),None)
                if item is None:raise ValueError(f'No detection chunk for camera {cam+1}, frame {local}')
                number=local-int(item['frame_start']);matrix=self.data.matrices[cam]
                if options.get('aruco'):
                    rows=self.table(item,'aruco').frame(number)
                    mm=project_points(np.c_[rows['X'],rows['Y']],matrix)*self.data.mm_per_pixel
                    for row,point in zip(rows[near_cup(mm,cup)],mm[near_cup(mm,cup)]):
                        output['aruco'].append(dict(camera=cam,id=int(row['Instance']),xy=cup_coordinates(point,cup,self.data.size)))
                if options.get('sleap'):
                    rows=self.table(item,'sleap').frame(number)
                    mm=project_points(np.c_[rows['X'],rows['Y']],matrix)*self.data.mm_per_pixel
                    for instance in np.unique(rows['Instance']):
                        take=rows['Instance']==instance
                        if not near_cup(mm[take],cup).any():continue
                        output['poses'].append(dict(camera=cam,instance=int(instance),bodypoints=rows['Bodypoint'][take].astype(int),
                            xy=cup_coordinates(mm[take],cup,self.data.size),scores=rows['Score_node'][take].astype(float)))
            except Exception as error:output['errors'].append(f'C{cam+1}: {error}')
        if options.get('track') or options.get('trail'):
            try:
                track=self.data.choices[index]['track']
                if track is not None:
                    path=self.data.root/'per_track'/f'TrackID_{track:04d}_all_single.parquet'
                    if track not in self.tracks:self.tracks[track]=TrackTable(path)
                    self.tracks.move_to_end(track)
                    while len(self.tracks)>2:self.tracks.popitem(last=False)[1].close()
                    rows,trail=self.tracks[track].frame(frame,round(self.data.fps*5) if options.get('trail') else 0)
                    if rows is not None and len(rows):
                        row=rows.iloc[0];position_source=str(row.PositionSource)
                        pose=rows.loc[rows.Bodypoint.ge(0)] if position_source!='aruco_only' else rows.iloc[:0]
                        output['track']=dict(source=position_source,aruco_id=int(row.ArucoID),
                            camera=int(row.SleapCam) if np.isfinite(row.SleapCam) else None,
                            aruco_camera=int(row.ArucoCam) if np.isfinite(row.ArucoCam) else None,
                            bodypoints=pose.Bodypoint.to_numpy(int),xy=cup_coordinates(pose[['X','Y']].to_numpy()*source['scale'],cup,self.data.size),
                            position=cup_coordinates(np.array([row.TrackX,row.TrackY])*source['scale'],cup,self.data.size))
                    if len(trail):
                        output['trail']=np.c_[trail[:,0],cup_coordinates(trail[:,1:]*source['scale'],cup,self.data.size)]
            except Exception as error:output['errors'].append('Track: '+str(error))
        return output

    def ownership(self,index,camera):
        key=index,camera
        if key not in self.geometry:
            plans,needed,_=self.data.geometry(index)
            plans=[p for p in plans if p['camera'] in needed]
            if camera is not None:
                p=next(p for p in self.data.geometry(index)[0] if p['camera']==camera)
                self.geometry[key]={camera:p['valid']}
            else:
                winner=np.stack([p['weight'] for p in plans]).argmax(axis=0)
                self.geometry[key]={p['camera']:(winner==i)&p['valid'] for i,p in enumerate(plans)}
        return self.geometry[key]

    def close(self):
        for table in list(self.tables.values())+list(self.tracks.values()):table.close()
        self.tables.clear();self.tracks.clear()


def draw_overlay(image, payload, options, masks=None):
    """Draw on a copy of a decoded frame; unavailable nodes never get invented."""
    from PIL import Image,ImageDraw,ImageFont,ImageChops
    picture=Image.fromarray(image).convert('RGBA')
    if payload is None:return picture.convert('RGB')
    font=ImageFont.truetype('DejaVuSans.ttf',12)
    def text(draw,point,label,color):
        if np.isfinite(point).all():draw.text(tuple(point+np.array([5,-16])),label,fill=color,font=font,stroke_width=2,stroke_fill='#191919')
    def skeleton(draw,pose,color):
        points={int(bp):xy for bp,xy in zip(pose['bodypoints'],pose['xy']) if np.isfinite(xy).all()}
        scores=dict(zip(pose['bodypoints'],pose.get('scores',np.ones(len(pose['bodypoints'])))))
        minimum=float(options.get('min_score',0)) if 'scores' in pose else 0
        points={bp:xy for bp,xy in points.items() if np.isnan(scores[bp]) or scores[bp]>=minimum}
        for a,b in payload['edges']:
            if a in points and b in points:draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=2)
        for bp,xy in points.items():
            x,y=xy;box=[x-3,y-3,x+3,y+3]
            draw.ellipse(box,fill=color if scores[bp]>=.2 else None,outline=color,width=2)
            if options.get('names'):text(draw,xy,payload['names'][bp] if bp<len(payload['names']) else str(bp),color)
        return points
    tags=payload['aruco'] if options.get('aruco') else []
    poses=payload['poses'] if options.get('sleap') else []
    cameras=sorted({p['camera'] for p in tags+poses})
    for camera in cameras:
        layer=Image.new('RGBA',picture.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
        for item in tags:
            if item['camera']!=camera:continue
            x,y=item['xy'];draw.line([(x-6,y),(x+6,y)],fill=RAW_TAG,width=2);draw.line([(x,y-6),(x,y+6)],fill=RAW_TAG,width=2)
            text(draw,item['xy'],f"C{camera+1} ID{item['id']}",RAW_TAG)
        for pose in poses:
            if pose['camera']!=camera:continue
            points=skeleton(draw,pose,RAW_POSE)
            if points:text(draw,np.mean(list(points.values()),axis=0),f"C{camera+1} S{pose['instance']}",RAW_POSE)
        if masks is not None and not options.get('all_cameras'):
            mask=masks.get(camera,np.zeros(image.shape[:2],bool))
            layer.putalpha(ImageChops.multiply(layer.getchannel('A'),Image.fromarray(mask.astype(np.uint8)*255)))
        picture=Image.alpha_composite(picture,layer)
    draw=ImageDraw.Draw(picture)
    if options.get('trail'):
        trail=payload['trail']
        for a,b in zip(trail[:-1],trail[1:]):
            if b[0]-a[0]==1 and np.isfinite(np.r_[a,b]).all():draw.line([tuple(a[1:]),tuple(b[1:])],fill=TRACK_POSITION,width=2)
    track=payload['track']
    if options.get('track') and track:
        skeleton(draw,track,TRACK_POSE)
        if np.isfinite(track['position']).all():
            x,y=track['position'];draw.polygon([(x,y-7),(x+7,y),(x,y+7),(x-7,y)],outline=TRACK_POSITION,width=2)
    return picture.convert('RGB')


def overlay_summary(payload):
    if not payload:return 'Detection overlays off'
    if payload.get('gap'):return 'No video or detections during the unrecorded gap'
    requested=payload.get('requested',dict(aruco=True,sleap=True,track=True))
    tags=f"{len(payload['aruco'])} ArUco detections" if requested.get('aruco') else 'ArUco overlay off'
    poses=f"{len(payload['poses'])} SLEAP instances" if requested.get('sleap') else 'SLEAP overlay off'
    result=f'Raw: {tags} · {poses}'
    track=payload['track']
    if track:
        finite=np.isfinite(track['xy']).all(axis=1);nodes=set(track['bodypoints'][finite])
        missing=[name for i,name in enumerate(payload['names']) if i not in nodes]
        camera=track['camera'];source=track['source'].replace('_',' + ') if track['source']=='aruco_sleap' else track['source'].replace('_',' ')
        result+=f" | Track ID{track['aruco_id']}: {source}, {len(nodes)}/{len(payload['names'])} pose nodes"
        if camera is not None and camera>=0:result+=f' (C{camera+1})'
        if missing:result+=' | Missing: '+', '.join(missing)
    else:result+=' | No assigned track at this frame' if requested.get('track') or requested.get('trail') else ' | Track overlay off'
    if payload['errors']:result+=' | UNAVAILABLE: '+'; '.join(payload['errors'])
    return result
