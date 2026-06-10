import time

import logging
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.processor.pipeline import IdentityProcessorStep, RobotProcessorPipeline, ObservationProcessorStep
from lerobot.processor.converters import observation_to_transition, transition_to_observation
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_robot_ur5e import UR5e, UR5eConfig
from lerobot_teleoperator_ur5e import UR5eTeleopKB, UR5eTeleopConfig
from mock_camera import MockCameraConfig

import numpy as np


logging.basicConfig(level=logging.WARNING)


def parse_camera_source(value: Any) -> Any:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def parse_joint_list(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{field_name} must be a list of 6 joint values.")
    return [float(joint) for joint in value]


def build_joint_action(joint_values: list[float]) -> dict[str, float]:
    action = {
        f"joint_{idx + 1}.pos": float(joint)
        for idx, joint in enumerate(joint_values)
    }
    action["gripper_position"] = 1.0
    return action


def build_gripper_action(is_open: bool) -> dict[str, float]:
    return {"gripper_position": 0.0 if is_open else 1.0}


def build_camera_config(camera_cfg: dict[str, Any], fps: int,
                        enabled: bool) -> CameraConfig:
    if not enabled:
        return MockCameraConfig(
            fps=fps,
            width=camera_cfg.get("width", 640),
            height=camera_cfg.get("height", 480),
            color_mode=ColorMode.RGB,
        )

    return OpenCVCameraConfig(
        index_or_path=parse_camera_source(camera_cfg["source"]),
        fps=fps,
        width=camera_cfg.get("width", 640),
        height=camera_cfg.get("height", 480),
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation(camera_cfg.get("rotation", 0)),
    )


def print_keymap(teleop: UR5eTeleopKB) -> None:
    print("\n===== Keymap =====")
    for line in teleop.keymap_lines():
        print(line)
    print("==================\n")


class ImageResizeProcessorStep(ObservationProcessorStep):
    """Simple image resizing processor step."""
    
    def __init__(self, resize_size: tuple[int, int] | None = None):
        self.resize_size = resize_size
    
    def observation(self, observation: dict) -> dict:
        if self.resize_size is None:
            return observation
        
        import cv2
        import torch
        from torchvision.transforms import functional as F
        
        new_observation = dict(observation)
        for key in observation:
            if "image" not in key:
                continue
            
            image = observation[key]
            
            # Resize if needed
            if isinstance(image, np.ndarray):
                if image.ndim >= 2 and tuple(image.shape[:2]) != self.resize_size:
                    target_h, target_w = self.resize_size
                    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    new_observation[key] = image
            elif isinstance(image, torch.Tensor) and image.shape[-2:] != self.resize_size:
                image = F.resize(image, self.resize_size)
                new_observation[key] = image
        
        return new_observation
    
    def get_config(self) -> dict[str, Any]:
        return {"resize_size": self.resize_size}
    
    def transform_features(self, features):
        return features


def make_robot_observation_processor_with_resize(
    storage_size: tuple[int, int] | None = None,
) -> RobotProcessorPipeline:
    """Create observation processor with optional image resizing.
    
    Args:
        storage_size: Tuple of (height, width) to resize images to, or None for no resizing
    
    Returns:
        RobotProcessorPipeline with image resizing step if storage_size is provided
    """
    steps = [IdentityProcessorStep()]
    
    if storage_size is not None:
        steps.append(ImageResizeProcessorStep(resize_size=storage_size))
    
    robot_observation_processor = RobotProcessorPipeline(
        steps=steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def update_observation_image_feature_shapes(
    obs_features: dict[str, dict], storage_height: int, storage_width: int
) -> dict[str, dict]:
    for key, ft in obs_features.items():
        if key.startswith("observation.images.") and isinstance(ft, dict):
            ft["shape"] = (storage_height, storage_width, 3)
    return obs_features


class RecordConfig:
    def __init__(self, cfg: dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time_cfg = cfg["time"]
        camera_cfg = cfg["cameras"]
        robot_cfg = cfg["robot"]
        teleop_cfg = cfg["teleop"]

        self.repo_id: str = cfg["repo_id"]
        self.fps: int = int(cfg.get("fps", 15))

        self.robot_ip: str = robot_cfg["ip"]
        self.enable_gripper: bool = bool(robot_cfg.get("enable_gripper", True))
        self.gripper_port: int = int(robot_cfg.get("gripper_port", 63352))
        self.gripper_open: int = int(robot_cfg.get("gripper_open", 0))
        self.gripper_close: int = int(robot_cfg.get("gripper_close", 255))
        self.gripper_force: int = int(robot_cfg.get("gripper_force", 255))
        self.gripper_init_open: bool = bool(
            robot_cfg.get("gripper_init_open", False)
        )
        self.gripper: dict[str, Any] = robot_cfg.get("gripper", {
            "choice": "robotiq",
            "robotiq": {
                "ip": self.robot_ip,
                "port": self.gripper_port,
                "force": self.gripper_force,
                "open": self.gripper_open,
                "close": self.gripper_close,
            },
        })
        self.servo_speed: float = float(robot_cfg.get("servo_speed", 0.1))
        self.servo_accel: float = float(robot_cfg.get("servo_accel", 0.1))
        self.servo_lookahead_time: float = float(
            robot_cfg.get("servo_lookahead_time", 0.1)
        )
        self.servo_gain: int = int(robot_cfg.get("servo_gain", 500))

        self.teleop_port: str = teleop_cfg["port"]
        self.teleop_joint_coef: list[float] = teleop_cfg.get(
            "joint_coef", [1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
        )
        self.resync_key: str = teleop_cfg.get("resync_key", "s")
        self.resync_settle_seconds: float = float(
            teleop_cfg.get("resync_settle_seconds", 0.2)
        )
        self.gripper_trigger_threshold: float = float(
            teleop_cfg.get("gripper_trigger_threshold", 400.0)
        )
        self.teleop_use_gripper: bool = bool(
            teleop_cfg.get("use_gripper", self.enable_gripper)
        )
        self.keyboard_linear_step: float = float(
            teleop_cfg.get("keyboard_linear_step", 0.01)
        )
        self.keyboard_angular_step: float = float(
            teleop_cfg.get("keyboard_angular_step", 0.08726646259971647)
        )
        self.keyboard_gripper_toggle_key: str = teleop_cfg.get(
            "keyboard_gripper_toggle_key", "g"
        )

        self.num_episodes: int = int(task.get("num_episodes", 1))
        self.display: bool = bool(task.get("display", True))
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = bool(task.get("resume", False))
        self.keyboard_init_joint: list[float] = parse_joint_list(
            task.get(
                "keyboard_init_joint",
                [
                    -1.8511,
                    -1.0227,
                    1.9796,
                    -0.9966,
                    1.2703,
                    -1.5707963267948966,
                ],
            ),
            "task.keyboard_init_joint",
        )
        self.close_gripper_before_recording: bool = bool(
            task.get("close_gripper_before_recording", False)
        )

        self.episode_time_sec: int = int(time_cfg.get("episode_time_sec", 60))
        self.reset_time_sec: int = int(time_cfg.get("reset_time_sec", 10))
        self.save_meta_period: int = int(time_cfg.get("save_meta_period", 1))

        self.enable_cameras: bool = bool(camera_cfg.get("enable", True))
        self.storage_width: int = int(camera_cfg.get("storage_width", 854))
        self.storage_height: int = int(camera_cfg.get("storage_height", 480))
        self.wrist_camera: dict[str, Any] = camera_cfg["wrist"]
        self.exterior_camera: dict[str, Any] = camera_cfg["exterior"]

        self.push_to_hub: bool = bool(storage.get("push_to_hub", False))


def main(record_cfg: RecordConfig):
    wrist_image_cfg = build_camera_config(record_cfg.wrist_camera,
                                          record_cfg.fps,
                                          record_cfg.enable_cameras)
    exterior_image_cfg = build_camera_config(record_cfg.exterior_camera,
                                             record_cfg.fps,
                                             record_cfg.enable_cameras)

    camera_config = {
        "wrist_image": wrist_image_cfg,
        "exterior_image": exterior_image_cfg,
    }
    teleop_config = UR5eTeleopConfig(
        port=record_cfg.teleop_port,
        robot_ip=record_cfg.robot_ip,
        enable_gripper=record_cfg.enable_gripper,
        use_gripper=record_cfg.teleop_use_gripper,
        joint_coef=record_cfg.teleop_joint_coef,
        resync_key=record_cfg.resync_key,
        resync_settle_seconds=record_cfg.resync_settle_seconds,
        gripper_trigger_threshold=record_cfg.gripper_trigger_threshold,
        keyboard_linear_step=record_cfg.keyboard_linear_step,
        keyboard_angular_step=record_cfg.keyboard_angular_step,
        keyboard_gripper_toggle_key=record_cfg.keyboard_gripper_toggle_key,
    )
    robot_config = UR5eConfig(
        robot_ip=record_cfg.robot_ip,
        enable_gripper=record_cfg.enable_gripper,
        gripper=record_cfg.gripper,
        gripper_init_open=record_cfg.gripper_init_open,
        servo_speed=record_cfg.servo_speed,
        servo_accel=record_cfg.servo_accel,
        servo_lookahead_time=record_cfg.servo_lookahead_time,
        servo_gain=record_cfg.servo_gain,
        control_mode="tcp",
        control_period_s=1.0 / record_cfg.fps,
        cameras=camera_config,
    )

    robot = UR5e(robot_config)
    teleop = UR5eTeleopKB(teleop_config)

    action_features = hw_to_dataset_features(teleop.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features,
                                          "observation",
                                          use_video=True)
    obs_features = update_observation_image_feature_shapes(
        obs_features, record_cfg.storage_height, record_cfg.storage_width
    )
    dataset_features = {**action_features, **obs_features}

    if record_cfg.resume:
        dataset = LeRobotDataset(record_cfg.repo_id)
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer()
        sanity_check_dataset_robot_compatibility(dataset, robot,
                                                 record_cfg.fps,
                                                 dataset_features)
    else:
        dataset = LeRobotDataset.create(
            repo_id=record_cfg.repo_id,
            fps=record_cfg.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    dataset.meta.metadata_buffer_size = record_cfg.save_meta_period

    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    teleop_action_processor, robot_action_processor, robot_observation_processor_base = (
        make_default_processors())
    
    # Override robot_observation_processor with image resizing
    robot_observation_processor = make_robot_observation_processor_with_resize(
        storage_size=(record_cfg.storage_height, record_cfg.storage_width)
    )

    robot.connect()
    teleop.connect()
    print_keymap(teleop)

    episode_idx = 0
    while episode_idx < record_cfg.num_episodes and not events[
            "stop_recording"]:
        log_say(
            f"Recording episode {episode_idx + 1} of {record_cfg.num_episodes}",
            play_sounds=False,
        )

        print(f"========== Episode:{episode_idx} ==========")

        while True:
            
            print("机械臂正在移动到初始位姿...\n")
            robot.send_joint_action(record_cfg.keyboard_init_joint, move_slow=True)

            robot.send_action(build_gripper_action(record_cfg.gripper_init_open))
            time.sleep(0.5)

            input("请在确认机械臂移动到初始位姿后，按 Enter 继续...\n")

            if record_cfg.close_gripper_before_recording:
                if record_cfg.enable_gripper:
                    input("你已启动录制前关闭夹爪，请按 Enter 闭合夹爪...\n")
                    robot.send_action({"gripper_position": 1.0})
                else:
                    logging.warning(
                        "close_gripper_before_recording is enabled, but gripper control is disabled."
                    )

            input("按 Enter 开始录制...\n")
            print("请进行遥操作...\n")

            record_loop(
                robot=robot,
                events=events,
                fps=record_cfg.fps,
                teleop=teleop,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=record_cfg.episode_time_sec,
                single_task=record_cfg.task_description,
                display_data=record_cfg.display,
            )

            if events["rerecord_episode"]:
                print("丢弃缓存，重新录制当前轨迹...")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            else:
                if events["stop_recording"]:
                    print("\n停止录制, 丢弃当前 episode 并退出...")
                else:
                    print("保存当前 Episode 中...")
                    dataset.save_episode()
                    print("保存当前 Episode 完毕...")
                break

        episode_idx += 1

    log_say("Stop recording", play_sounds=False)
    robot.disconnect()
    teleop.disconnect()
    dataset.finalize()
    if record_cfg.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    with open(Path(__file__).parent / "config" / "cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    main(record_cfg)
