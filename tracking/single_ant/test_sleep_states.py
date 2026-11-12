import numpy as np
from tracking.single_ant.sleep_states import classify, sustained_high


def test_missing_quiet_time_stays_ambiguous_and_original_assignments_win():
    observed=np.array([0,1,2,-1,-1,-1,-1],np.int8)
    speed=np.array([1,1,0,1,0,0,1],bool);antenna=np.array([1,1,0,0,1,0,1],bool)
    recorded=np.array([1,1,1,1,1,1,0],bool)
    category,evidence=classify(observed,speed,antenna,recorded)
    np.testing.assert_array_equal(category,[1,1,0,0,0,-1,-1])
    np.testing.assert_array_equal(evidence,[1,1,1,2,4,0,0])


def test_exclusion_needs_repeated_current_motion_and_resets_at_boundaries():
    manifest=dict(n_frames=20,recording_segments=[dict(frame_offset=0,num_frames=8),dict(frame_offset=10,num_frames=10)])
    values=np.array([0,9,0,0,0,0,0,9,np.nan,np.nan,9,9,9,9,9,9,np.nan,0,0,0])
    high=sustained_high(values,5,manifest)
    assert not high[:15].any()  # Full trailing window restarts at frame 10.
    assert high[15] and not high[16:].any() # No carryover into missing/quiet frames.


def test_sleep_requires_observed_s0_or_s1_even_with_available_slow_motion():
    category,evidence=classify(np.full(10,-1),np.zeros(10,bool),np.zeros(10,bool),np.ones(10,bool))
    assert (category==-1).all() and not evidence.any()
