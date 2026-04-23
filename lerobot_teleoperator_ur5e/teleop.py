import logging
import select
import sys
import termios
import threading
import time
import tty
from typing import Any

import numpy as np
from rtde_receive import RTDEReceiveInterface

try:
    import alicia_d_sdk
except ImportError:  # pragma: no cover - hardware dependency
    alicia_d_sdk = None

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_teleop import UR5eTeleopConfig

KEYBOARD_POLL_SECONDS = 0.1
logger = logging.getLogger(__name__)


def _format_python_list(values: np.ndarray) -> str:
    return np.array2string(
        values,
        precision=4,
        separator=", ",
        max_line_width=100000,
    )


class KeyMonitor:
    def __init__(self, keys: str | list[str]):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = {key.lower() for key in keys}
        self._pending_requests = {key: False for key in self.keys}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._fd = None
        self._old_settings = None
        self.enabled = False

    def start(self) -> None:
        if not sys.stdin.isatty():
            logger.warning(
                "stdin is not a TTY, teleop hotkeys %s are disabled", sorted(self.keys)
            )
            return

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.enabled and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self.enabled = False

    def consume_request(self, key: str) -> bool:
        with self._lock:
            normalized_key = key.lower()
            requested = self._pending_requests.get(normalized_key, False)
            if normalized_key in self._pending_requests:
                self._pending_requests[normalized_key] = False
            return requested

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], KEYBOARD_POLL_SECONDS)
            if not ready:
                continue

            key = sys.stdin.read(1).lower()
            if key in self.keys:
                with self._lock:
                    self._pending_requests[key] = True


class TeleoperatorDevice:
    def __init__(self, port: str, gripper_trigger_threshold: float):
        self.port = port
        self.gripper_trigger_threshold = gripper_trigger_threshold
        self.robot = None
        self.prev_q = np.zeros(6)
        self.compensation = np.zeros(6)
        self.prev_gripper_level = 1000.0

    def connect(self) -> None:
        if alicia_d_sdk is None:
            raise ImportError(
                "alicia_d_sdk is required for the master-arm teleoperator"
            )

        logger.info("Connecting teleoperator on %s", self.port)
        self.robot = alicia_d_sdk.create_robot(port=self.port)
        version = self.robot.get_robot_state("version")
        if version:
            logger.info(
                "Teleoperator connected: serial=%s hw=%s fw=%s",
                version.get("serial_number"),
                version.get("hardware_version"),
                version.get("firmware_version"),
            )

        initial_q = self.get_joint_positions()
        if initial_q is not None:
            self.prev_q = initial_q.copy()

        initial_gripper = self.get_gripper_level()
        if initial_gripper is not None:
            self.prev_gripper_level = initial_gripper

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def get_joint_positions(self) -> np.ndarray | None:
        try:
            q = self.robot.get_robot_state("joint")
            return None if q is None else np.array(q, dtype=float)
        except Exception as exc:
            logger.warning("Failed to read teleoperator joints: %s", exc)
            return None

    def get_smoothed_joint_positions(self) -> np.ndarray:
        current_q = self.get_joint_positions()
        if current_q is None:
            return self.prev_q + self.compensation

        variation = current_q - self.prev_q
        self.compensation[variation > np.pi] -= 2.0 * np.pi
        self.compensation[variation < -np.pi] += 2.0 * np.pi
        self.prev_q = current_q.copy()
        return current_q + self.compensation

    def reset_smoothing(self) -> None:
        current_q = self.get_joint_positions()
        if current_q is None:
            return
        self.prev_q = current_q.copy()
        self.compensation = np.zeros(6)

    def get_gripper_level(self) -> float | None:
        try:
            level = self.robot.get_robot_state("gripper")
            return None if level is None else float(level)
        except Exception as exc:
            logger.warning("Failed to read teleoperator gripper: %s", exc)
            return None

    def consume_gripper_toggle(self) -> bool:
        current_level = self.get_gripper_level()
        if current_level is None:
            return False

        triggered = (
            self.prev_gripper_level > self.gripper_trigger_threshold
            and current_level < self.gripper_trigger_threshold
        )
        self.prev_gripper_level = current_level
        return triggered


class UR5eTeleop(Teleoperator):
    config_class = UR5eTeleopConfig
    name = "ur5e_master_arm"

    def __init__(self, config: UR5eTeleopConfig):
        super().__init__(config)
        self.config = config
        self.device = TeleoperatorDevice(config.port, config.gripper_trigger_threshold)
        self.state_print_key = "/"
        self.key_monitor = KeyMonitor([config.resync_key, self.state_print_key])
        self.joint_coef = np.array(config.joint_coef, dtype=float)
        self.robot_receiver = None
        self.delta: np.ndarray | None = None
        self.sync_enabled = True
        self.gripper_position = 0.0 if not config.enable_gripper else 1.0

    @property
    def action_features(self) -> dict:
        features = {f"joint_{idx}.pos": float for idx in range(1, 7)}
        features["gripper_position"] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.device.robot is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Teleoperator is already connected.")
        self.device.connect()
        self.robot_receiver = RTDEReceiveInterface(self.config.robot_ip)
        self.key_monitor.start()

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return

    def get_robot_joint_positions(self) -> np.ndarray | None:
        try:
            q = self.robot_receiver.getActualQ()
            return None if q is None else np.array(q, dtype=float)
        except Exception as exc:
            logger.warning("Failed to read UR5e joints for teleop sync: %s", exc)
            return None

    def get_robot_tcp_pose(self) -> np.ndarray | None:
        try:
            tcp_pose = self.robot_receiver.getActualTCPPose()
            return None if tcp_pose is None else np.array(tcp_pose, dtype=float)
        except Exception as exc:
            logger.warning("Failed to read UR5e TCP pose for state print: %s", exc)
            return None

    def calibrate_delta(self) -> None:
        robot_q = self.get_robot_joint_positions()
        if robot_q is None:
            raise RuntimeError(
                "Robot joint state is not available for teleop calibration"
            )

        teleoperator_q = self.device.get_joint_positions()
        if teleoperator_q is None:
            raise RuntimeError(
                "Teleoperator joint state is not available for teleop calibration"
            )

        self.delta = robot_q - self.joint_coef * teleoperator_q
        logger.info(
            "Teleop delta recalibrated: %s", np.array2string(self.delta, precision=4)
        )

    def _handle_sync_toggle(self) -> None:
        if not self.key_monitor.consume_request(self.config.resync_key):
            return

        if self.sync_enabled:
            self.sync_enabled = False
            logger.info("Teleop sync paused")
            return

        logger.info("Recalibrating and resuming teleop sync")
        self.device.reset_smoothing()
        time.sleep(self.config.resync_settle_seconds)
        self.calibrate_delta()
        self.sync_enabled = True
        logger.info("Teleop sync resumed")

    def _handle_state_print(self) -> None:
        if not self.key_monitor.consume_request(self.state_print_key):
            return

        self.print_current_robot_state()

    def print_current_robot_state(self) -> None:
        robot_q = self.get_robot_joint_positions()
        tcp_pose = self.get_robot_tcp_pose()

        print("\n===== Current Robot State =====")
        if robot_q is None:
            print("joint: unavailable")
        else:
            print(f"joint = {_format_python_list(robot_q)}")

        if tcp_pose is None:
            print("tcp:   unavailable")
        else:
            print(f"tcp = {_format_python_list(tcp_pose)}")
        print("================================\n")

    def keymap_lines(self) -> list[str]:
        lines = [
            "操作说明:",
            "  右方向键: 结束并保存当前 Episode，在显示“请使用遥操臂进行遥操作...”时按才生效",
            "  左方向键: 丢弃数据，重新录制当前 Episode，在显示“请使用遥操臂进行遥操作...”时按才生效",
            "  Esc: 立即结束录制，在显示“请使用遥操臂进行遥操作...”时按才生效",
            f"  {self.state_print_key}: 打印当前 ur5e 的状态信息（关节状态和tcp状态）",
            "遥操操作:",
            "  移动遥操臂即可"
        ]
        if self.config.enable_gripper and self.config.use_gripper:
            lines.append("  遥操臂扳机: 切换夹爪开关状态")
        return lines

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "UR5eTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._handle_sync_toggle()
        self._handle_state_print()

        if (
            self.config.enable_gripper
            and self.config.use_gripper
            and self.device.consume_gripper_toggle()
        ):
            self.gripper_position = 0.0 if self.gripper_position >= 0.5 else 1.0

        robot_q = self.get_robot_joint_positions()
        if robot_q is None:
            raise RuntimeError(
                "Robot joint state is required before teleop actions can be computed"
            )

        if self.delta is None:
            self.calibrate_delta()

        if self.sync_enabled:
            teleoperator_q = self.device.get_smoothed_joint_positions()
            target_joint = self.joint_coef * teleoperator_q + self.delta
        else:
            target_joint = robot_q

        action_dict = {
            f"joint_{idx + 1}.pos": float(target_joint[idx]) for idx in range(6)
        }
        action_dict["gripper_position"] = (
            self.gripper_position if self.config.enable_gripper else 0.0
        )
        return action_dict

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self.key_monitor.stop()
        if self.robot_receiver is not None:
            self.robot_receiver.disconnect()
            self.robot_receiver = None
        self.device.disconnect()
