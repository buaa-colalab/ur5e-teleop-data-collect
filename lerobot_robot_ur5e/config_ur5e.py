from dataclasses import dataclass, field
from typing import Any, Literal

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("ur5e_robot")
@dataclass
class UR5eConfig(RobotConfig):
    robot_ip: str = "192.168.119.3"
    enable_gripper: bool = True
    gripper: dict[str, Any] = field(default_factory=lambda: {
        "choice": "robotiq",
        "qb": {
            "port": "/dev/ttyUSB0",
            "device_id": 1,
            "open": (0, 0),
            "close": (8000, 0),
        },
        "dh": {
            "port": "/dev/ttyUSB0",
            "force": 100,
            "open": 1000,
            "close": 0,
        },
        "robotiq": {
            "ip": None,
            "port": 63352,
            "speed": 255,
            "force": 255,
            "open": 0,
            "close": 255,
        },
    })
    # Legacy Robotiq-only fields kept for older scripts/configs.
    gripper_port: int = 63352
    gripper_open: int = 0
    gripper_close: int = 255
    gripper_force: int = 255
    gripper_init_open: bool = False
    servo_speed: float = 0.1
    servo_accel: float = 0.1
    servo_lookahead_time: float = 0.1
    servo_gain: int = 500
    control_mode: Literal["joint", "tcp"] = "joint"
    control_period_s: float = 1 / 15
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
