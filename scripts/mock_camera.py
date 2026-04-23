from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.cameras import Camera, CameraConfig, ColorMode


@CameraConfig.register_subclass("mock")
@dataclass
class MockCameraConfig(CameraConfig):
    color_mode: ColorMode = ColorMode.RGB


class MockCamera(Camera):
    def __init__(self, config: MockCameraConfig):
        super().__init__(config)
        self.color_mode = config.color_mode
        self._is_connected = False
        self._frame = self._build_frame()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def connect(self, warmup: bool = True) -> None:
        self._is_connected = True

    def _build_frame(self) -> np.ndarray:
        height = int(self.height or 480)
        width = int(self.width or 640)
        self.height = height
        self.width = width

        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)

        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = x[None, :]
        frame[..., 1] = y[:, None]
        frame[..., 2] = 127
        return frame

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        requested_color_mode = color_mode or self.color_mode
        if requested_color_mode == self.color_mode:
            return self._frame.copy()
        return self._frame[..., ::-1].copy()

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self.read()

    def disconnect(self) -> None:
        self._is_connected = False
