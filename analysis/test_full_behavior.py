"""Scientific invariants: alignment, exact contacts, and gap-aware windows."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from analysis.full_behavior_extract import align_body, count_contacts, MM_PER_PX
from analysis.full_behavior_states import segment_windows
from analysis.full_behavior_sleep import sleep_from_windows, build, contact_history_audit
from analysis.full_behavior_profiles import state_sums


class FullBehaviorTests(unittest.TestCase):
    def test_zero_trip_ant_keeps_zero_rate_and_missing_duration(self):
        from analysis.trip_phenotyping_utils import compute_trip_investment_confidence
        ants=pd.DataFrame(dict(side=['right']*2,track_id=[1,2],track_name=['one','two'],frame_min=[0,0],frame_max=[86399,86399]))
        trips=pd.DataFrame(dict(track_name=['one'],duration_minutes=[2.]))
        result=compute_trip_investment_confidence(ants,trips,fps=1,n_bootstrap=10).set_index('track_id')
        self.assertEqual(result.loc[2,'completed_trips_per_day'],0)
        self.assertGreater(result.loc[2,'completed_trips_per_day_ci_high'],0)
        self.assertTrue(np.isnan(result.loc[2,'median_trip_minutes']))
        self.assertEqual(result.loc[1,'median_trip_minutes'],2)

    def test_contact_audit_separates_missing_history_and_pre_onset_state_changes(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.npy';np.save(path,[1,1,1,1,1,1,0,0,-1,1])
            tracks=pd.DataFrame(dict(side=['right'],track_id=[1],cluster_id=['C0'],state_path=[path],frame_min=[0]))
            bouts=pd.DataFrame(dict(side=['right']*3,ant_a=[1]*3,ant_b=[2]*3,is_new_onset=[True]*3,start_frame=[6,8,10]))
            row=contact_history_audit(tracks,bouts,fps=2,prior_seconds=2).iloc[0]
            self.assertEqual(row.new_contact_onsets,3);self.assertEqual(row.full_prior_labels,2)
            self.assertEqual(row.prior_contains_unknown,1);self.assertEqual(row.full_prior_sleep,1)
            self.assertEqual(row.guarded_prior_sleep,2);self.assertEqual(row.guarded_only,1)

    def test_state_means_use_observed_samples_and_duration_not_window_count(self):
        # The long and short S0 windows have unequal weight. A huge unassigned
        # observation and missing gap must not contaminate the mean skeleton.
        pose=np.full((12,20),np.nan)
        pose[:4]=np.arange(4)[:,None];pose[4]=1e9
        pose[6:8]=10;pose[9:12]=-2
        raw=np.zeros((3,31));raw[:,24:]=np.array([2,8,5])[:,None]
        count,shape,motion=state_sums(pose,np.array([0,6,9]),np.array([4,2,3]),
                                    np.array([0,0,1]),raw,2,batch_size=1)
        np.testing.assert_array_equal(count,[6,3])
        np.testing.assert_allclose(shape[0]/count[0],26/6)
        np.testing.assert_allclose(shape[1]/count[1],-2)
        np.testing.assert_allclose(motion[0]/count[0],4)
        np.testing.assert_allclose(motion[1]/count[1],5)

    def test_body_alignment_removes_rigid_motion_preserves_gaster(self):
        skeleton=np.array([[0,0],[12,0],[-20,0],[-45,4],[16,5],[24,12],[33,12],
                           [16,-5],[24,-12],[33,-12]],float)[None]
        theta=1.1;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        a,_,valid=align_body(skeleton,20*MM_PER_PX)
        b,_,_=align_body(skeleton@rotation.T+1000,20*MM_PER_PX)
        np.testing.assert_allclose(a,b,atol=1e-10);self.assertTrue(valid.all())
        bent=skeleton.copy();bent[:,3,1]+=10
        c,_,_=align_body(bent,20*MM_PER_PX)
        self.assertNotEqual(a[0,3,1],c[0,3,1])
        np.testing.assert_allclose(a[:,4:],c[:,4:])

    def test_concurrent_contacts_are_deduplicated_and_never_joined(self):
        # The same partner remains present at frames 0 and 2, but frame 1 is 0.
        frames=np.array([0,0,0,2,2]);a=np.array([1,2,1,1,1]);b=np.array([2,1,3,2,2])
        counts=count_contacts(frames,a,b,np.array([1,2]),4)
        np.testing.assert_array_equal(counts,[[2,0,1,0],[1,0,1,0]])

    def test_windows_never_cross_missing_or_camera_changes(self):
        valid=np.ones(30,bool);valid[8:10]=False
        camera=np.zeros(30,int);camera[21:]=1
        starts,lengths=segment_windows(valid,camera,width=12,minimum=3)
        np.testing.assert_array_equal(starts,[0,10,21])
        np.testing.assert_array_equal(lengths,[8,11,9])
        for s,n in zip(starts,lengths):
            self.assertTrue(valid[s:s+n].all());self.assertEqual(np.ptp(camera[s:s+n]),0)

    def test_window_tails_do_not_create_fake_zero_rate_samples(self):
        starts,lengths=segment_windows(np.ones(25,bool),np.zeros(25),width=12,minimum=3)
        np.testing.assert_array_equal(starts,[0,12]);np.testing.assert_array_equal(lengths,[12,12])
        starts,lengths=segment_windows(np.zeros(5,bool),np.zeros(5))
        self.assertEqual(len(starts),0)

    def test_state_sleep_preserves_gaps_and_censors_bouts(self):
        state,bouts=sleep_from_windows(np.array([0,5,8]),np.array([3,3,3]),
                                      np.array([0,0,1]),24,[0])
        np.testing.assert_array_equal(state[:6],1)
        np.testing.assert_array_equal(state[6:10],-1)
        np.testing.assert_array_equal(state[10:16],1)
        np.testing.assert_array_equal(state[16:22],0)
        self.assertTrue(bouts.iloc[0].right_censored)
        self.assertFalse(bouts.iloc[1].right_censored)
        self.assertTrue(bouts.left_censored.all())

    def test_duration_filter_does_not_call_censored_quiet_awake(self):
        state,_=sleep_from_windows(np.array([0,5,8]),np.array([3,3,3]),
                                  np.array([0,0,1]),24,[0],minimum_seconds=10)
        np.testing.assert_array_equal(state[:16],-1)
        np.testing.assert_array_equal(state[16:22],0)

    def test_downstream_sleep_uses_states_without_existing_sleep_labels(self):
        from analysis.sleep_motion_analysis_utils import load_sleep_label_tracks
        with TemporaryDirectory() as directory:
            root=Path(directory)/'run';block=Path(directory)/'block'
            (root/'models').mkdir(parents=True)
            # Both states meet the motion-based candidate rule, but the default
            # sleep definition is explicitly S0. S1 must remain non-sleep.
            raw=np.zeros((2,31));raw[:,24:28]=[.01,.01,.07,.07]
            np.savez(root/'models/dictionary.npz',raw=raw,state=[0,1])
            pd.DataFrame(dict(ant=['right:001','right:002'],side=['right']*2,
                track_id=[1,2],track_name=['one.parquet','two.parquet'],n_frames=[24]*2,
                selected=[True,False])).to_csv(root/'inventory.csv',index=False)
            for name in ('one','two'):
                folder=block/'stitched/sleep_motion/per_track'/name;folder.mkdir(parents=True)
                (folder/'sleep_motion_metadata.json').write_text(json.dumps(dict(frame_min=2,frame_max=21)))
            folder=root/'features/right_001';folder.mkdir(parents=True)
            np.save(folder/'start.npy',[0,5,8]);np.save(folder/'length.npy',[3,3,3])
            folder=root/'assignments/right_001';folder.mkdir(parents=True);np.save(folder/'state.npy',[0,0,1])
            output=build(root,block)
            tracks=load_sleep_label_tracks(output)
            self.assertEqual(tracks.frame_min.tolist(),[2,2])
            a=np.load(tracks.state_path.iloc[0]);b=np.load(tracks.state_path.iloc[1])
            self.assertEqual(len(a),20);self.assertEqual(int((a==1).sum()),10)
            np.testing.assert_array_equal(a[14:20],0)
            self.assertEqual(json.loads((output/'definition.json').read_text())['quiet_states'],[0])
            self.assertTrue((b==-1).all())
            timestamp=Path(tracks.metadata_path.iloc[0]).stat().st_mtime_ns
            build(root,block)
            self.assertEqual(Path(tracks.metadata_path.iloc[0]).stat().st_mtime_ns,timestamp)


if __name__=='__main__':unittest.main()
