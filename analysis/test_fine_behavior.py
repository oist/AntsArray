import unittest
import numpy as np

from analysis.fine_behavior_extract import fill_short_gaps, sample_starts, measurements, FPS, START, MM_PER_PX
from analysis.fine_behavior_features import spectrum, window_features, encode, HZ


class FineBehaviorTest(unittest.TestCase):
    def test_body_speed_unsigned_and_antenna_invariant_to_rigid_motion(self):
        skeleton=np.array([[0,0],[12,0],[-20,0],[-45,0],[16,5],[24,12],[33,12],[16,-5],[24,-12],[33,-12]],float)
        time=np.arange(120)/FPS
        for sign,angle in ((1,0),(-1,0),(1,np.pi/2)):
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            moving=skeleton[None,:,:]+np.column_stack([sign*time/MM_PER_PX,np.zeros(len(time))])[:,None,:]
            xy=(moving@rotation.T)[None,:,:,:]
            camera=np.zeros(xy.shape[:2])
            result=measurements(xy,xy[:,:,0],camera,camera,.32,np.zeros(12),np.eye(12))
            signals=result['signals'][0,1:]
            np.testing.assert_allclose(signals[:,8],1,atol=1e-6)
            np.testing.assert_allclose(signals[:,9:12],0,atol=1e-6)

    def test_short_fills_do_not_cross_camera_or_long_gap(self):
        x=np.arange(20.)[None,:,None,None]*np.ones((1,1,1,2))
        camera=np.zeros((1,20))
        x[:,3:6]=np.nan;x[:,8:12]=np.nan;x[:,15:17]=np.nan
        camera[:,16:]=1
        y,filled=fill_short_gaps(x,camera)
        np.testing.assert_allclose(y[0,3:6,0,0],[3,4,5])
        self.assertTrue(np.isnan(y[0,8:12]).all())
        self.assertTrue(np.isnan(y[0,15:17]).all())
        self.assertEqual(filled.sum(),3)

    def test_sampling_stays_inside_each_hour(self):
        starts=sample_starts(7)
        offsets=starts-START-np.arange(48)*3600*FPS
        self.assertTrue(((offsets>=0)&(offsets+60*FPS<=3600*FPS)).all())
        np.testing.assert_array_equal(starts,sample_starts(7))

    def test_wavelets_recover_frequency_and_reject_gap_camera(self):
        time=np.arange(720)/HZ
        scores=np.tile(np.sin(2*np.pi*1.2*time)[None,:,None],(1,1,4))
        camera=np.zeros((1,720))
        power=spectrum(scores,camera)
        mean=np.nanmean(power,axis=(0,1)).reshape(4,6)
        self.assertTrue((mean.argmax(axis=1)==2).all())
        scores[:,300]=np.nan;camera[:,500:]=1
        power=spectrum(scores,camera)
        self.assertTrue(np.isnan(power[0,299:302]).all())
        self.assertTrue(np.isnan(power[0,499:502]).all())
        self.assertTrue(np.isfinite(power[0,150]).all())

    def test_ordered_trajectory_distinguishes_equal_mean_postures(self):
        n=120
        signals=np.zeros((2,n,13));signals[0,:,0]=np.arange(n);signals[1,:,0]=-np.arange(n)
        power=np.ones((2,n,24));cams=np.zeros((2,n));frames=np.tile(np.arange(n)*2,(2,1))
        partners=np.zeros((2,n));onsets=np.array([20,24,30])
        base,trajectory,power,valid,centers=window_features(signals,power,cams,frames,partners,onsets,1.)
        np.testing.assert_allclose(trajectory[0],-trajectory[1])
        self.assertTrue(valid.all())
        # First centered window spans raw frames [12,37), inclusive data endpoints.
        self.assertEqual(base[0,0,-1],3)

    def test_training_scales_ignore_heldout_values(self):
        rng=np.random.default_rng(11)
        base=rng.random((3,30,16));trajectory=rng.normal(size=(3,30,40));power=rng.random((3,30,24))
        data=dict(base=base,trajectory=trajectory,spectrum=power)
        train=np.arange(20)
        x,norm=encode(data,0,'spectrum',train)
        data['base'][:,20:]*=100;data['spectrum'][:,20:]*=100
        xx,other=encode(data,0,'spectrum',train)
        self.assertEqual(norm,other)
        np.testing.assert_allclose(x[:20],xx[:20])
        self.assertEqual(x.shape[1],41)

    def test_short_support_does_not_require_long_wavelets(self):
        signals=np.zeros((1,120,13));power=np.full((1,120,24),np.nan)
        cameras=np.zeros((1,120));frames=np.arange(120)[None,:]*2;partners=np.zeros((1,120))
        arguments=(signals,power,cameras,frames,partners,np.array([],int),.5)
        self.assertFalse(window_features(*arguments)[3].any())
        self.assertTrue(window_features(*arguments,require_spectrum=False)[3].all())


if __name__=='__main__':unittest.main()
