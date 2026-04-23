from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lerobot_teleoperator_ur5e")
@dataclass(kw_only=True)
class UR5eTeleopConfig(TeleoperatorConfig):
    port: str = "/dev/ttyUSB0"
    robot_ip: str = "192.168.1.184"
    enable_gripper: bool = True
    use_gripper: bool = True
    keyboard_linear_step: float = 0.01
    keyboard_angular_step: float = 0.08726646259971647
    keyboard_gripper_toggle_key: str = "g"
    joint_coef: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
    )
    resync_key: str = "s"
    resync_settle_seconds: float = 0.2
    gripper_trigger_threshold: float = 400.0
