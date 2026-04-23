import select
import sys
import termios
import threading
import time
import tty

import alicia_d_sdk
import numpy as np
import rtde_control
import rtde_receive


UR5E_IP = "192.168.31.123"
TELEOPERATOR_PORT = "/dev/ttyACM0"

CONTROL_FREQUENCY_HZ = 15
JOINT_COEF = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=float)

SERVO_SPEED = 0.1
SERVO_ACCEL = 0.1
SERVO_LOOKAHEAD_TIME = 0.1
SERVO_GAIN = 500

RESYNC_SETTLE_SECONDS = 0.2
KEYBOARD_POLL_SECONDS = 0.1


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class KeyMonitor:
    def __init__(self):
        self._toggle_sync_requested = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._fd = None
        self._old_settings = None
        self.enabled = False

    def start(self) -> None:
        if not sys.stdin.isatty():
            log("stdin is not a TTY, hotkey 's' is disabled")
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

    def consume_toggle_sync_request(self) -> bool:
        with self._lock:
            requested = self._toggle_sync_requested
            self._toggle_sync_requested = False
            return requested

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], KEYBOARD_POLL_SECONDS)
            if not ready:
                continue

            key = sys.stdin.read(1)
            if key.lower() == "s":
                with self._lock:
                    self._toggle_sync_requested = True


class TeleoperatorDevice:
    def __init__(self, port: str):
        self.port = port
        self.robot = None
        self.prev_q = np.zeros(6)
        self.compensation = np.zeros(6)
        self._connect()

    def _connect(self) -> None:
        log(f"connecting teleoperator on {self.port}")
        self.robot = alicia_d_sdk.create_robot(port=self.port)
        version = self.robot.get_robot_state("version")
        if version:
            log(
                "teleoperator connected: "
                f"serial={version.get('serial_number')} "
                f"hw={version.get('hardware_version')} "
                f"fw={version.get('firmware_version')}"
            )

        initial_q = self.get_joint_positions()
        if initial_q is not None:
            self.prev_q = np.array(initial_q, dtype=float)

    def get_joint_positions(self):
        try:
            q = self.robot.get_robot_state("joint")
            if q is None:
                return None
            return np.array(q, dtype=float)
        except Exception as exc:
            log(f"failed to read teleoperator joints: {exc}")
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

    def disconnect(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.disconnect()
        except Exception as exc:
            log(f"failed to disconnect teleoperator: {exc}")


class UR5eTeleopRobot:
    def __init__(self):
        self.rtde_c = None
        self.rtde_r = None
        self._connect()

    def _connect(self) -> None:
        log(f"connecting UR5e on {UR5E_IP}")
        self.rtde_c = rtde_control.RTDEControlInterface(UR5E_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR5E_IP)

    def get_joint_positions(self):
        try:
            q = self.rtde_r.getActualQ()
            if q is None:
                return None
            return np.array(q, dtype=float)
        except Exception as exc:
            log(f"failed to read UR5e joints: {exc}")
            return None

    def servo_j(self, target_q: np.ndarray, dt: float) -> None:
        self.rtde_c.servoJ(
            target_q.tolist(),
            SERVO_SPEED,
            SERVO_ACCEL,
            dt,
            SERVO_LOOKAHEAD_TIME,
            SERVO_GAIN,
        )

    def servo_stop(self) -> None:
        try:
            self.rtde_c.servoStop()
        except Exception as exc:
            log(f"failed to stop servoJ: {exc}")

    def disconnect(self) -> None:
        try:
            self.servo_stop()
            if self.rtde_c is not None:
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            if self.rtde_r is not None:
                self.rtde_r.disconnect()
        except Exception as exc:
            log(f"failed to disconnect UR5e: {exc}")


def calibrate_delta(
    robot: UR5eTeleopRobot, teleoperator: TeleoperatorDevice
) -> np.ndarray:
    ur5e_q = robot.get_joint_positions()
    teleoperator_q = teleoperator.get_joint_positions()

    if ur5e_q is None:
        ur5e_q = np.zeros(6)
    if teleoperator_q is None:
        teleoperator_q = np.zeros(6)

    delta = ur5e_q - JOINT_COEF * teleoperator_q
    log(f"calibrated delta: {np.array2string(delta, precision=4)}")
    return delta


def run() -> int:
    dt = 1.0 / CONTROL_FREQUENCY_HZ
    teleoperator = None
    robot = None
    key_monitor = KeyMonitor()
    sync_enabled = True

    try:
        teleoperator = TeleoperatorDevice(TELEOPERATOR_PORT)
        robot = UR5eTeleopRobot()
        delta = calibrate_delta(robot, teleoperator)

        key_monitor.start()
        log("teleop sync started")
        log("press 's' once to pause sync")
        log(
            "adjust the teleoperator pose, then press 's' again to recalibrate and resume"
        )
        log("press Ctrl-C to exit")

        next_tick = time.monotonic()
        while True:
            if key_monitor.consume_toggle_sync_request():
                if sync_enabled:
                    sync_enabled = False
                    robot.servo_stop()
                    log("sync paused")
                else:
                    log("recalibrating and resuming sync")
                    teleoperator.reset_smoothing()
                    time.sleep(RESYNC_SETTLE_SECONDS)
                    delta = calibrate_delta(robot, teleoperator)
                    sync_enabled = True
                    next_tick = time.monotonic()
                    log("sync resumed")

            if not sync_enabled:
                time.sleep(KEYBOARD_POLL_SECONDS)
                next_tick = time.monotonic()
                continue

            teleoperator_q = teleoperator.get_smoothed_joint_positions()
            target_joint = JOINT_COEF * teleoperator_q + delta
            robot.servo_j(target_joint, dt)

            next_tick += dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        log("stopped by user")
        return 0
    except Exception as exc:
        log(f"fatal error: {exc}")
        return 1
    finally:
        key_monitor.stop()
        if robot is not None:
            robot.disconnect()
        if teleoperator is not None:
            teleoperator.disconnect()


if __name__ == "__main__":
    raise SystemExit(run())
