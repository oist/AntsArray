"""Identify redundant whole-pose observations without merging distinct ants."""
from __future__ import annotations

import numpy as np


def duplicate_groups(candidates, nodes_for_instance, cameras, tag_ids, *,
                     association_distance, same_camera_only=False):
    """Complete-link groups; six landmarks, axis and identity must agree.

    Distances use the tracker's input units. Tolerances also scale with the
    smaller anatomical axis, so bounds appropriate for large single ants do not
    merge nearby small colony ants. No coordinates are averaged or imputed.
    """
    groups=[]
    if len(candidates)<2:return [list(candidates)] if candidates else []
    limit=float(association_distance)

    def equivalent(left,right):
        a,b=int(left[1]),int(right[1])
        if same_camera_only and cameras.get(a)!=cameras.get(b):return False
        ta,tb=tag_ids.get(a),tag_ids.get(b)
        if ta is not None and tb is not None and ta!=tb:return False
        na,nb=nodes_for_instance(a),nodes_for_instance(b)
        common=sorted(k for k in set(na)&set(nb)
                      if np.isfinite(na[k]).all() and np.isfinite(nb[k]).all())
        if len(common)<6 or 0 not in common:return False
        axis_node=2 if 2 in common else 1 if 1 in common else None
        if axis_node is None:return False
        ax=np.asarray(na[0])-na[axis_node];bx=np.asarray(nb[0])-nb[axis_node]
        la,lb=float(np.linalg.norm(ax)),float(np.linalg.norm(bx))
        if not np.isfinite([la,lb]).all() or min(la,lb)<=1e-6:return False
        scale=min(la,lb)
        if scale/max(la,lb)<.7 or np.dot(ax,bx)/(la*lb)<np.cos(np.deg2rad(20)):
            return False
        if np.linalg.norm(np.asarray(na[0])-nb[0])>min(limit,.5*scale):return False
        distance=np.asarray([np.linalg.norm(np.asarray(na[k])-nb[k]) for k in common])
        return bool(np.isfinite(distance).all()
                    and (distance<=min(limit*.35/1.44,.25*scale)).sum()>=6
                    and np.median(distance)<=min(limit*.25/1.44,.15*scale))

    for candidate in candidates:
        for group in groups:
            if all(equivalent(candidate,other) for other in group):
                group.append(candidate);break
        else:groups.append([candidate])
    return groups
