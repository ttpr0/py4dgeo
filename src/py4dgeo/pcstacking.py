from ast import List
from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.util import (
    as_double_precision,
    MemoryPolicy,
    Py4DGeoError,
    make_contiguous,
    memory_policy_is_minimum,
)

import abc
import logging
import numpy as np
import typing
import laspy

import _py4dgeo


logger = logging.getLogger("py4dgeo")

def pointcloud_stacking(epoch: Epoch, radius: float, max_distance: float) -> Epoch:
    """Stack point cloud of an epoch.

    Parameters
    ----------
    epoch : Epoch
        The epoch containing the point cloud to be stacked.
    radius : float
        The radius used for stacking.
    max_distance : float
        The maximum distance for stacking.
    """
    # ensure covariances are present
    if epoch.covariances is None:
        raise Py4DGeoError("Covariances are required for point cloud stacking.")

    return _py4dgeo.pointcloud_stacking(
        epoch, radius, max_distance
    )
