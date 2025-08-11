from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTermCfg, ActionTerm
from isaaclab_hsr.tasks.manager_based.isaaclab_hsr.mdp.actions import hsr_actions

##
# HSR actions.
##

@configclass
class HSRBaseVelocityControlCfg(ActionTermCfg):
    """Configuration for the HSR holonomic action term.

    See :class:`HSRHolonomicAction` for more details.
    """

    class_type: type[ActionTerm] = hsr_actions.HSRBaseVelocityControl

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    
@configclass
class HSRGripperActionCfg(ActionTermCfg):
    """Configuration for the HSR gripper action term.

    See :class:`HSRGripperAction` for more details.
    """

    class_type: type[ActionTerm] = hsr_actions.HSRGripperAction
    
@configclass
class HSRBBinaryGripperActionCfg(ActionTermCfg):
    """Configuration for the HSR binary gripper action term.

    See :class:`HSRBBinaryGripperAction` for more details.
    """
    
    class_type: type[ActionTerm] = hsr_actions.HSRBBinaryGripperAction