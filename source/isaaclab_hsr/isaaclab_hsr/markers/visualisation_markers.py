import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

RED_ARROW_NEGATIVE_Y_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"/workspace/isaaclab/source/standalone/hsrb/arrow_negative_y.usd",
            scale=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)
"""Configuration for the red arrow marker (along negative y-direction)."""

RED_ARROW_Y_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"/workspace/isaaclab/source/standalone/hsrb/arrow_y.usd",
            scale=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)
"""Configuration for the red arrow marker (along y-direction)."""