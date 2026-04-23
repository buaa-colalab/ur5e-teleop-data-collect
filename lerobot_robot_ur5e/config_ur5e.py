from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("ur5e_robot")
@dataclass
class UR5eConfig(RobotConfig):
    robot_ip: str = "192.168.1.184"
    enable_gripper: bool = True
    gripper_port: str = "/dev/ur5e_left_gripper"
    gripper_open: int = 0
    gripper_close: int = 1000
    servo_speed: float = 0.1
    servo_accel: float = 0.1
    servo_lookahead_time: float = 0.1
    servo_gain: int = 500
    control_period_s: float = 1 / 15
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
