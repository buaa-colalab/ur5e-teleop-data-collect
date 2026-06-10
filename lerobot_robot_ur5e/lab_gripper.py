from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_GRIPPER_CONFIG: dict[str, Any] = {
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
}


class LabGripper:
    """Unified 0.0=open to 1.0=closed gripper interface."""

    def __init__(self, config: Mapping[str, Any] | None = None, robot_ip: str | None = None):
        merged = deepcopy(DEFAULT_GRIPPER_CONFIG)
        self._merge_config(merged, dict(config or {}))
        self.config = merged
        choice = str(self.config.get("choice", "robotiq")).lower()
        self.choice = {
            "ag95": "dh",
            "qb_softhand": "qb",
            "softhand": "qb",
            "soft_hand": "qb",
        }.get(choice, choice)
        if self.choice not in {"robotiq", "dh", "qb"}:
            raise ValueError(
                f"Unsupported gripper choice={self.choice!r}. "
                "Expected one of: 'robotiq', 'dh', 'qb'."
            )

        self.robot_ip = robot_ip
        self._device = None
        self._last_command: Any = None
        self._last_position = 0.0
        self._connected = False

    @staticmethod
    def _merge_config(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                target[key].update(value)
            else:
                target[key] = value

    @classmethod
    def from_legacy_config(
        cls,
        *,
        robot_ip: str,
        gripper_port: int,
        gripper_open: int,
        gripper_close: int,
        gripper_force: int,
    ) -> "LabGripper":
        return cls(
            {
                "choice": "robotiq",
                "robotiq": {
                    "ip": robot_ip,
                    "port": gripper_port,
                    "force": gripper_force,
                    "open": gripper_open,
                    "close": gripper_close,
                },
            },
            robot_ip=robot_ip,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        print(f"\n[GRIPPER] Initializing {self.choice} gripper...")
        if self.choice == "robotiq":
            self._connect_robotiq()
        elif self.choice == "dh":
            self._connect_dh()
        else:
            self._connect_qb()
        self._connected = True
        self._last_position = self.get_position()
        print("[GRIPPER] Gripper initialized successfully.\n")

    def disconnect(self) -> None:
        if self._device is None:
            self._connected = False
            return

        try:
            if self.choice == "qb":
                self._device.stop(True)
                self._device.close()
            elif self.choice == "robotiq":
                self._device.disconnect()
            else:
                for method_name in ("close", "disconnect"):
                    method = getattr(self._device, method_name, None)
                    if callable(method):
                        method()
                        break
                serial = getattr(self._device, "ser", None)
                if serial is not None and hasattr(serial, "close"):
                    serial.close()
        finally:
            self._device = None
            self._connected = False

    def open(self) -> None:
        self.set_position(0.0)

    def close(self) -> None:
        self.set_position(1.0)

    def set_position(self, value: float) -> None:
        if self._device is None:
            return

        normalized = self._clamp(value)
        command = self._normalized_to_command(normalized)
        if command == self._last_command:
            return

        if self.choice == "robotiq":
            cfg = self._choice_config()
            self._device.move(
                int(command),
                speed=int(cfg.get("speed", 255)),
                force=int(cfg.get("force", 255)),
            )
        elif self.choice == "dh":
            self._device.set_pos(int(command))
        else:
            self._device.set_additive_synergies(*command)

        self._last_command = command
        self._last_position = normalized

    def get_position(self) -> float:
        if self._device is None:
            return self._last_position

        raw_position = None
        try:
            if self.choice == "robotiq":
                raw_position = self._device.get_current_position()
            elif self.choice == "qb":
                raw_position = self._device.get_synergies()
            else:
                raw_position = self._read_dh_position()
        except Exception:
            raw_position = None

        if raw_position is None:
            return self._last_position

        self._last_position = self._command_to_normalized(raw_position)
        return self._last_position

    def _connect_robotiq(self) -> None:
        from .robotiq_gripper import RobotiqGripper

        cfg = self._choice_config()
        hostname = cfg.get("ip") or self.robot_ip
        if not hostname:
            raise ValueError("Robotiq gripper requires 'ip' or robot_ip.")

        gripper = RobotiqGripper()
        gripper.connect(hostname=hostname, port=int(cfg.get("port", 63352)))
        gripper.activate(auto_calibrate=False)
        self._device = gripper

    def _connect_dh(self) -> None:
        from pyDHgripper import AG95

        cfg = self._choice_config()
        gripper = AG95(port=str(cfg.get("port", "/dev/ttyUSB0")))
        gripper.set_force(int(cfg.get("force", 100)))
        self._device = gripper

    def _connect_qb(self) -> None:
        import qbdevice_py

        cfg = self._choice_config()
        hand = qbdevice_py.SoftHand2Controller(
            str(cfg.get("port", "/dev/ttyUSB0")),
            device_id=int(cfg.get("device_id", 1)),
        )
        hand.validate_connection()
        hand.enable_motors(True)
        self._device = hand

    def _read_dh_position(self) -> Any:
        for method_name in ("get_pos", "get_position", "get_current_position"):
            method = getattr(self._device, method_name, None)
            if callable(method):
                return method()
        return None

    def _choice_config(self) -> dict[str, Any]:
        return dict(self.config[self.choice])

    def _normalized_to_command(self, value: float) -> Any:
        cfg = self._choice_config()
        open_value = cfg["open"]
        close_value = cfg["close"]

        if self.choice == "qb":
            open_tuple = self._as_number_tuple(open_value)
            close_tuple = self._as_number_tuple(close_value)
            if len(open_tuple) != len(close_tuple):
                raise ValueError("qb gripper open/close commands must have the same length.")
            return tuple(
                int(round(o + (c - o) * value))
                for o, c in zip(open_tuple, close_tuple, strict=True)
            )

        return int(round(float(open_value) + (float(close_value) - float(open_value)) * value))

    def _command_to_normalized(self, raw_value: Any) -> float:
        cfg = self._choice_config()
        open_value = cfg["open"]
        close_value = cfg["close"]

        if self.choice == "qb":
            raw_tuple = self._as_number_tuple(raw_value)
            open_tuple = self._as_number_tuple(open_value)
            close_tuple = self._as_number_tuple(close_value)
            return self._scalar_to_normalized(raw_tuple[0], open_tuple[0], close_tuple[0])

        raw_scalar = self._first_number(raw_value)
        return self._scalar_to_normalized(raw_scalar, float(open_value), float(close_value))

    @staticmethod
    def _scalar_to_normalized(value: float, open_value: float, close_value: float) -> float:
        span = close_value - open_value
        if span == 0:
            return 0.0
        return LabGripper._clamp((value - open_value) / span)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _as_number_tuple(value: Any) -> tuple[float, ...]:
        if isinstance(value, Mapping):
            value = list(value.values())
        elif isinstance(value, str):
            stripped = value.strip().strip("()[]")
            value = [part.strip() for part in stripped.split(",") if part.strip()]
        elif not isinstance(value, Iterable):
            value = [value]

        numbers = tuple(float(item) for item in value)
        if not numbers:
            raise ValueError("Expected at least one numeric gripper value.")
        return numbers

    @staticmethod
    def _first_number(value: Any) -> float:
        return LabGripper._as_number_tuple(value)[0]
