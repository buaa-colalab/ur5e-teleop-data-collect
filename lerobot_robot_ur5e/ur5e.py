import logging
import threading
import time
from typing import Any

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

try:
    from pyDHgripper import AG95
except ImportError:  # pragma: no cover - hardware dependency
    AG95 = None

from .config_ur5e import UR5eConfig

logger = logging.getLogger(__name__)


def _skew(vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ],
        dtype=float,
    )


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)

    axis = rotvec / theta
    skew_axis = _skew(axis)
    return (
        np.eye(3)
        + np.sin(theta) * skew_axis
        + (1.0 - np.cos(theta)) * (skew_axis @ skew_axis)
    )


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-12:
        return np.zeros(3, dtype=float)

    if np.pi - theta < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0))
        axis = axis.astype(float)
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
            axis[2] = np.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        elif axis[1] > 1e-6:
            axis[0] = np.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
            axis[2] = np.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
        else:
            axis[0] = np.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
            axis[1] = np.copysign(axis[1], matrix[1, 2] + matrix[2, 1])

        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = axis / norm
        return axis * theta

    scale = theta / (2.0 * np.sin(theta))
    return scale * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    )


def _rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    matrix = _rotvec_to_matrix(rotvec)
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    cos_pitch = float(np.cos(pitch))

    if abs(cos_pitch) < 1e-8:
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        yaw = 0.0
    else:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))

    return np.array([roll, pitch, yaw], dtype=float)


def _rpy_to_rotvec(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _matrix_to_rotvec(_rpy_to_matrix(roll, pitch, yaw))


class UR5e(Robot):
    config_class = UR5eConfig
    name = "ur5e"

    def __init__(self, config: UR5eConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.config = config
        self._is_connected = False
        self._arm = {}
        self._gripper = None
        self._prev_observation = None
        self._gripper_command = 1.0
        self._gripper_lock = threading.Lock()
        self._stop_gripper_reader = threading.Event()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.name} is already connected.")

        self._arm["rtde_r"], self._arm["rtde_c"] = self._check_ur5e_connection(
            self.config.robot_ip)
        if self.config.enable_gripper:
            self._gripper = self._check_gripper_connection(
                self.config.gripper_port)
            self._gripper_command = self._raw_gripper_to_normalized(
                self.config.gripper_close)
            self._start_gripper_state_reader()
        else:
            self._gripper = None
            self._gripper_command = 0.0

        print("\n===== Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            print(f"[CAM] {cam_name} connected successfully.")
        print("===== Cameras Initialized Successfully =====\n")

        self.is_connected = True
        print(
            f"[INFO] {self.name} env initialization completed successfully.\n")

    def _check_gripper_connection(self, port: str):
        print("\n[GRIPPER] Initializing AG95 gripper...")
        if AG95 is None:
            raise ImportError(
                "pyDHgripper.AG95 is required for the AG95 gripper integration"
            )
        gripper = AG95(port)
        gripper.init_feedback()
        gripper.set_force(100)
        print("[GRIPPER] Gripper initialized successfully.\n")
        return gripper

    def _check_ur5e_connection(self, robot_ip: str):
        try:
            print("\n[ROBOT] Connecting to UR5e robot...")
            rtde_r = RTDEReceiveInterface(robot_ip)
            rtde_c = RTDEControlInterface(robot_ip)

            joint_positions = rtde_r.getActualQ()
            if joint_positions is not None and len(joint_positions) == 6:
                formatted_joints = [round(j, 4) for j in joint_positions]
                print(f"[ROBOT] Current joint positions: {formatted_joints}")
                print("[ROBOT] UR5e connected successfully.\n")
            else:
                print(
                    "[ERROR] Failed to read joint positions. Check connection or remote control mode."
                )
        except Exception as exc:
            print("[ERROR] Failed to connect to UR5e robot.")
            print(f"Exception: {exc}\n")
            raise

        return rtde_r, rtde_c

    def _start_gripper_state_reader(self):
        threading.Thread(target=self._read_gripper_state, daemon=True).start()

    def _read_gripper_state(self):
        if self._gripper is None:
            return
        self._gripper.pos = None
        last_command = None
        while not self._stop_gripper_reader.is_set():
            with self._gripper_lock:
                command = self._gripper_command
            raw_command = self._normalized_gripper_to_raw(command)
            if raw_command != last_command:
                self._gripper.set_pos(val=raw_command)
                last_command = raw_command
            self._gripper.pos = self._gripper.read_pos()
            time.sleep(0.02)

    def _normalized_gripper_to_raw(self, value: float) -> int:
        clamped = max(0.0, min(1.0, float(value)))
        span = self.config.gripper_close - self.config.gripper_open
        return int(round(self.config.gripper_open + span * clamped))

    def _raw_gripper_to_normalized(self, value: float | None) -> float:
        if value is None:
            return 0.0
        span = self.config.gripper_close - self.config.gripper_open
        if span == 0:
            return 0.0
        return max(0.0,
                   min(1.0, (float(value) - self.config.gripper_open) / span))

    def _extract_joint_position_action(
            self, action: dict[str, Any]) -> list[float] | None:
        joint_keys = [f"joint_{idx}.pos" for idx in range(1, 7)]
        if all(key in action for key in joint_keys):
            return [float(action[key]) for key in joint_keys]

        joint_position = action.get("joint_position")
        if joint_position is None:
            return None
        if len(joint_position) != 6:
            raise ValueError(
                f"joint_position must have 6 values, got {len(joint_position)}"
            )
        return [float(value) for value in joint_position]

    def _extract_tcp_pose_action(self, action: dict[str, Any]) -> list[float] | None:
        position_keys = ["tcp_pose.x", "tcp_pose.y", "tcp_pose.z"]
        orientation_keys = [
            "tcp_pose.roll",
            "tcp_pose.pitch",
            "tcp_pose.yaw",
        ]

        if not all(key in action for key in position_keys + orientation_keys):
            return None

        x, y, z = (float(action[key]) for key in position_keys)
        roll, pitch, yaw = (float(action[key]) for key in orientation_keys)
        rotvec = _rpy_to_rotvec(roll, pitch, yaw)
        return [x, y, z, float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "tcp_pose.x": float,
            "tcp_pose.y": float,
            "tcp_pose.z": float,
            "tcp_pose.roll": float,
            "tcp_pose.pitch": float,
            "tcp_pose.yaw": float,
            "gripper_position": float,
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "gripper_position": float,
        }

    def send_action(self, action: dict[str, Any], move_slow=False) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_position = self._extract_joint_position_action(action)
        tcp_pose = self._extract_tcp_pose_action(action)
        if joint_position is not None:
            if not move_slow:
                t_start = self._arm["rtde_c"].initPeriod()
                self._arm["rtde_c"].servoJ(
                    joint_position,
                    self.config.servo_speed,
                    self.config.servo_accel,
                    self.config.control_period_s,
                    self.config.servo_lookahead_time,
                    self.config.servo_gain,
                )
                self._arm["rtde_c"].waitPeriod(t_start)
            else:
                t_start = self._arm["rtde_c"].initPeriod()
                self._arm["rtde_c"].servoJ(
                    joint_position,
                    0.01,
                    0.01,
                    2,
                    0.2,
                    200,
                )
                self._arm["rtde_c"].waitPeriod(t_start)
        elif tcp_pose is not None:
            if not move_slow:
                t_start = self._arm["rtde_c"].initPeriod()
                self._arm["rtde_c"].servoL(
                    tcp_pose,
                    self.config.servo_speed,
                    self.config.servo_accel,
                    self.config.control_period_s,
                    self.config.servo_lookahead_time,
                    self.config.servo_gain,
                )
                self._arm["rtde_c"].waitPeriod(t_start)
            else:
                t_start = self._arm["rtde_c"].initPeriod()
                self._arm["rtde_c"].servoL(
                    tcp_pose,
                    0.01,
                    0.01,
                    2.0,
                    0.2,
                    400,
                )
                self._arm["rtde_c"].waitPeriod(t_start)

        if "gripper_position" in action:
            if self.config.enable_gripper:
                with self._gripper_lock:
                    self._gripper_command = float(action["gripper_position"])

        return action

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_position = self._arm["rtde_r"].getActualQ()
        tcp_pose = np.array(self._arm["rtde_r"].getActualTCPPose(), dtype=float)
        tcp_rpy = _rotvec_to_rpy(tcp_pose[3:])

        obs_dict = {}
        for i in range(len(joint_position)):
            obs_dict[f"joint_{i + 1}.pos"] = joint_position[i]

        for i, axis in enumerate(["x", "y", "z"]):
            obs_dict[f"tcp_pose.{axis}"] = float(tcp_pose[i])

        for i, axis in enumerate(["roll", "pitch", "yaw"]):
            obs_dict[f"tcp_pose.{axis}"] = float(tcp_rpy[i])

        if self.config.enable_gripper and self._gripper is not None:
            obs_dict["gripper_position"] = self._raw_gripper_to_normalized(
                self._gripper.pos)
        else:
            obs_dict["gripper_position"] = 0.0

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        self._prev_observation = obs_dict
        return obs_dict

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        self._stop_gripper_reader.set()

        if self._arm:
            try:
                self._arm["rtde_c"].servoStop()
            except Exception:
                pass
            try:
                self._arm["rtde_c"].stopScript()
            except Exception:
                pass
            self._arm["rtde_c"].disconnect()
            self._arm["rtde_r"].disconnect()

        if self._gripper is not None:
            try:
                self._gripper.close()
            except AttributeError:
                pass

        for cam in self.cameras.values():
            cam.disconnect()

        self.is_connected = False
        logger.info("[INFO] ===== All connections have been closed =====")

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return self.is_connected

    def configure(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
