"""Check S0 calibration weights, fresh labels beyond pose support, and causality."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from analysis import s0_motion_sleep as s0
from analysis import sleep_motion_utils as sm


class S0MotionTests(unittest.TestCase):
    def test_context_refresh_replaces_old_sleep_and_age_keeps_physical_arrays(self):
        old=dict(inside=np.array([1,1,0,1]),body_speed=np.array([1.,2.,3.,4.]),
            state=np.ones(4),sleep_age_seconds=np.full(4,999.))
        result=s0.replace_context_sleep(old,np.array([1,1,-1,0,1,1,1,1]),0,2,2)
        np.testing.assert_array_equal(result['state'],[1,0,1,1])
        np.testing.assert_array_equal(result['sleep_age_seconds'],[1,0,1,2])
        np.testing.assert_array_equal(result['inside'],old['inside'])
        np.testing.assert_array_equal(result['body_speed'],old['body_speed'])

    def test_calibration_weights_ants_not_observation_counts(self):
        sample=pd.DataFrame(dict(ant=['a']*100+['b']+['c'],state=[0]*101+[1],
            body=[1.]*100+[3.,1000.],antenna=[2.]*100+[4.,1000.]))
        self.assertEqual(s0.thresholds(sample),dict(body=3.,antenna=4.))

    def test_fresh_labels_extend_past_posture_gaps_and_ignore_future_motion(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';block=root/'block';work=root/'work';out=root/'labels'
            (source/'models').mkdir(parents=True);np.savez(source/'models/dictionary.npz',state=[0,1])
            pd.DataFrame([dict(ant='right:001',track_name='one.parquet',track_id=1,side='right',
                n_frames=2400,selected=True)]).to_csv(source/'inventory.csv',index=False)
            features=source/'features/right_001';features.mkdir(parents=True)
            np.save(features/'start.npy',[0,600]);np.save(features/'length.npy',[200,600])
            assignments=source/'assignments/right_001';assignments.mkdir(parents=True)
            np.save(assignments/'state.npy',[0,1])
            cache=block/'stitched/sleep_motion/per_track/one';cache.mkdir(parents=True)
            speed=np.full((2400,10),.1,np.float32);speed[:,4:]=.2;speed[1200:]=2
            np.save(cache/sm.SPEED_FILENAME,speed);np.save(cache/sm.FRAME_GAP_FILENAME,np.ones((2400,10),np.uint8))
            np.save(cache/sm.BODYPOINT_IDS_FILENAME,np.arange(10));np.save(cache/sm.FRAMES_FILENAME,np.arange(2400))
            (cache/sm.METADATA_FILENAME).write_text(json.dumps(dict(n_frames=2400,frame_min=0,frame_max=2399,
                fps=24.,speed_shape=[2400,10],cache_max_gap_frames=5,motion_version=sm.MOTION_VERSION)))
            s0.prepare(source,block,work,0);s0.calibrate(source,work,out);s0.label(source,work,out,0);s0.finalize(source,out)
            state=np.load(out/'per_track/one/sleep_state.npy')
            self.assertEqual(state[800],1)  # No classified posture at this time.
            self.assertEqual(state[1800],0);self.assertEqual(state[0],-1)
            body=np.load(work/'measurements/one/body.npy');ant=np.load(work/'measurements/one/antenna.npy')
            params=json.loads((out/'definition.json').read_text())['classifier_parameters']
            body[1200:]=100
            changed=sm.classify_body_antenna_quiet(body,ant,fps=24,
                body_speed_threshold_mm_s=params['body_speed_threshold_mm_s'],
                antenna_speed_threshold_mm_s=params['antenna_speed_threshold_mm_s'])['state']
            np.testing.assert_array_equal(changed[:1200],state[:1200])
            metadata=json.loads((cache/sm.METADATA_FILENAME).read_text())
            metadata.pop('motion_version')
            (cache/sm.METADATA_FILENAME).write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'Rebuild the motion cache'):
                s0.prepare(source,block,work,0)


if __name__=='__main__':unittest.main()
