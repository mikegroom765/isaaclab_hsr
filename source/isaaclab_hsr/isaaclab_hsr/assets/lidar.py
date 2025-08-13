# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Velodyne LiDAR sensors."""


from isaaclab.sensors import RayCasterCfg, patterns

##
# Configuration
##


HOKUYO_UST_20LX_RAYCASTER_CFG = RayCasterCfg(
    ray_alignment='base',
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(45, 315), horizontal_res=1.0 # actual value is 0.25, number of points is downsampled so is used 1.0
    ),
    debug_vis=True,
    max_distance=60,
)

"""Configuration for Hokuyo UST-20LX LiDAR as a :class:`RayCasterCfg`.

Reference: https://www.hokuyo-aut.jp/search/single.php?serial=167#spec
"""