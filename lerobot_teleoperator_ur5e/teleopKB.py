import logging
import os
import sys
from queue import Queue
from typing import Any

import numpy as np
from rtde_receive import RTDEReceiveInterface

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_teleop import UR5eTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as exc:  # pragma: no cover - environment dependent
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info("Could not import pynput: %s", exc)

logger = logging.getLogger(__name__)


def _format_python_list(values: np.ndarray) -> str:
    return np.array2string(
        values,
        precision=4,
        separator=", ",
        max_line_width=100000,
    )


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


def _matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    cos_pitch = float(np.cos(pitch))

    if abs(cos_pitch) < 1e-8:
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        yaw = 0.0
    else:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))

    return np.array([roll, pitch, yaw], dtype=float)


def _rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    return _matrix_to_rpy(_rotvec_to_matrix(rotvec))


def _axis_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)

    unit_axis = axis / axis_norm
    skew_axis = _skew(unit_axis)
    return (
        np.eye(3)
        + np.sin(angle) * skew_axis
        + (1.0 - np.cos(angle)) * (skew_axis @ skew_axis)
    )


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class UR5eTeleopKB(Teleoperator):
    config_class = UR5eTeleopConfig
    name = "ur5e_keyboard"

    def __init__(self, config: UR5eTeleopConfig):
        super().__init__(config)
        self.config = config
        self.event_queue = Queue()
        self.current_pressed: dict[str, bool] = {}
        self.listener = None
        self.robot_receiver = None
        self.gripper_position = 1.0 if config.enable_gripper else 0.0

    @property
    def action_features(self) -> dict:
        return {
            "tcp_pose.x": float,
            "tcp_pose.y": float,
            "tcp_pose.z": float,
            "tcp_pose.roll": float,
            "tcp_pose.pitch": float,
            "tcp_pose.yaw": float,
            "gripper_position": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self.robot_receiver is not None
            and self.listener is not None
            and self.listener.is_alive()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Keyboard teleop is already connected.")
        if not PYNPUT_AVAILABLE:
            raise RuntimeError(
                "pynput is required for keyboard teleoperation. "
                "Make sure a graphical keyboard listener is available."
            )

        self.robot_receiver = RTDEReceiveInterface(self.config.robot_ip)
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return

    def _normalize_key(self, key: Any) -> str | None:
        if not hasattr(key, "char") or key.char is None:
            return None
        return key.char.lower()

    def _on_press(self, key: Any) -> None:
        key_char = self._normalize_key(key)
        if key_char is not None:
            self.event_queue.put((key_char, True))

    def _on_release(self, key: Any) -> None:
        key_char = self._normalize_key(key)
        if key_char is not None:
            self.event_queue.put((key_char, False))

    def _drain_key_events(self) -> tuple[int, bool]:
        toggle_count = 0
        state_print_requested = False
        toggle_key = self.config.keyboard_gripper_toggle_key.lower()

        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            was_pressed = self.current_pressed.get(key_char, False)
            self.current_pressed[key_char] = is_pressed
            if key_char == toggle_key and is_pressed and not was_pressed:
                toggle_count += 1
            elif key_char == "/" and is_pressed and not was_pressed:
                state_print_requested = True

        return toggle_count, state_print_requested

    def keymap_lines(self) -> list[str]:
        lines = [
            "操作说明:",
            "  右方向键: 结束并保存当前 Episode，在显示“请进行遥操作...”时按才生效",
            "  左方向键: 丢弃数据，重新录制当前 Episode，在显示“请进行遥操作...”时按才生效",
            "  Esc: 立即结束录制，在显示“请进行遥操作...”时按才生效",
            "  /: 打印当前 ur5e 的状态信息（关节状态和tcp状态）",
            "遥操操作:",
            "  w / s: TCP x+ / x- (forward / backward)",
            "  a / d: TCP y+ / y- (left / right)",
            "  q / e: TCP z+ / z- (up / down)",
            "  u / j: tool local Rx + / -",
            "  i / k: tool local Ry + / -",
            "  o / l: tool local Rz + / -",
        ]
        if self.config.enable_gripper and self.config.use_gripper:
            lines.append(
                f"  {self.config.keyboard_gripper_toggle_key}: toggle gripper open/close"
            )
        return lines

    def get_robot_joint_positions(self) -> np.ndarray:
        if self.robot_receiver is None:
            raise DeviceNotConnectedError("Keyboard teleop robot receiver is not connected.")

        joint_positions = self.robot_receiver.getActualQ()
        if joint_positions is None:
            raise RuntimeError("Robot joint state is not available for keyboard teleop.")
        return np.array(joint_positions, dtype=float)

    def print_current_robot_state(self) -> None:
        joint_positions = self.get_robot_joint_positions()
        tcp_pose = self.get_robot_tcp_pose()
        tcp_rpy = _rotvec_to_rpy(tcp_pose[3:])

        print("\n===== Current Robot State =====")
        print(f"joint = {_format_python_list(joint_positions)}")
        print(f"tcp = {_format_python_list(tcp_pose)}")
        print(f"rpy = {_format_python_list(tcp_rpy)}")
        print("================================\n")

    def get_robot_tcp_pose(self) -> np.ndarray:
        if self.robot_receiver is None:
            raise DeviceNotConnectedError("Keyboard teleop robot receiver is not connected.")

        tcp_pose = self.robot_receiver.getActualTCPPose()
        if tcp_pose is None:
            raise RuntimeError("Robot TCP pose is not available for keyboard teleop.")
        return np.array(tcp_pose, dtype=float)

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "UR5eTeleopKB is not connected. You need to run `connect()` before `get_action()`."
            )

        toggle_count, state_print_requested = self._drain_key_events()
        if (
            self.config.enable_gripper
            and self.config.use_gripper
            and toggle_count % 2 == 1
        ):
            self.gripper_position = 0.0 if self.gripper_position >= 0.5 else 1.0

        if state_print_requested:
            self.print_current_robot_state()

        tcp_pose = self.get_robot_tcp_pose()
        current_rotation = _rotvec_to_matrix(tcp_pose[3:])

        linear_step = float(self.config.keyboard_linear_step)
        angular_step = float(self.config.keyboard_angular_step)

        delta_x = linear_step * (
            float(self.current_pressed.get("w", False))
            - float(self.current_pressed.get("s", False))
        )
        delta_y = linear_step * (
            float(self.current_pressed.get("a", False))
            - float(self.current_pressed.get("d", False))
        )
        delta_z = linear_step * (
            float(self.current_pressed.get("q", False))
            - float(self.current_pressed.get("e", False))
        )

        delta_roll = angular_step * (
            float(self.current_pressed.get("u", False))
            - float(self.current_pressed.get("j", False))
        )
        delta_pitch = angular_step * (
            float(self.current_pressed.get("i", False))
            - float(self.current_pressed.get("k", False))
        )
        delta_yaw = angular_step * (
            float(self.current_pressed.get("o", False))
            - float(self.current_pressed.get("l", False))
        )

        delta_rotation = (
            _axis_rotation_matrix(np.array([1.0, 0.0, 0.0]), delta_roll)
            @ _axis_rotation_matrix(np.array([0.0, 1.0, 0.0]), delta_pitch)
            @ _axis_rotation_matrix(np.array([0.0, 0.0, 1.0]), delta_yaw)
        )

        target_position = tcp_pose[:3] + np.array(
            [delta_x, delta_y, delta_z],
            dtype=float,
        )
        # Apply orientation increments in the tool-local frame so each key
        # consistently rotates around the end-effector's current x/y/z axes.
        target_rotation = current_rotation @ delta_rotation
        target_rpy = _matrix_to_rpy(target_rotation)
        target_rpy = np.array([_wrap_to_pi(angle) for angle in target_rpy], dtype=float)

        return {
            "tcp_pose.x": float(target_position[0]),
            "tcp_pose.y": float(target_position[1]),
            "tcp_pose.z": float(target_position[2]),
            "tcp_pose.roll": float(target_rpy[0]),
            "tcp_pose.pitch": float(target_rpy[1]),
            "tcp_pose.yaw": float(target_rpy[2]),
            "gripper_position": (
                self.gripper_position if self.config.enable_gripper else 0.0
            ),
        }

    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

        if self.robot_receiver is not None:
            self.robot_receiver.disconnect()
            self.robot_receiver = None
