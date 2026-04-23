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
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_robot_ur5e import UR5e, UR5eConfig
from lerobot_teleoperator_ur5e import UR5eTeleop, UR5eTeleopConfig
from mock_camera import MockCameraConfig


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
    action["gripper_position"] = 1000.0
    return action


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


def print_keymap(teleop: UR5eTeleop) -> None:
    print("\n===== Keymap =====")
    for line in teleop.keymap_lines():
        print(line)
    print("==================\n")


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
        self.gripper_port: str = robot_cfg["gripper_port"]
        self.gripper_open: int = int(robot_cfg.get("gripper_open", 0))
        self.gripper_close: int = int(robot_cfg.get("gripper_close", 1000))
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

        self.num_episodes: int = int(task.get("num_episodes", 1))
        self.display: bool = bool(task.get("display", True))
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = bool(task.get("resume", False))
        self.teleop_init_joint: list[float] = parse_joint_list(
            task.get(
                "teleop_init_joint",
                [
                    -2.6177242437945765,
                    -2.009850164453024,
                    2.4285362402545374,
                    -0.4312353891185303,
                    1.307356834411621,
                    -1.5594866911517542,
                ],
            ),
            "task.teleop_init_joint",
        )
        self.close_gripper_before_recording: bool = bool(
            task.get("close_gripper_before_recording", False)
        )

        self.episode_time_sec: int = int(time_cfg.get("episode_time_sec", 60))
        self.reset_time_sec: int = int(time_cfg.get("reset_time_sec", 10))
        self.save_meta_period: int = int(time_cfg.get("save_meta_period", 1))

        self.enable_cameras: bool = bool(camera_cfg.get("enable", True))
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
    )
    robot_config = UR5eConfig(
        robot_ip=record_cfg.robot_ip,
        enable_gripper=record_cfg.enable_gripper,
        gripper_port=record_cfg.gripper_port,
        gripper_open=record_cfg.gripper_open,
        gripper_close=record_cfg.gripper_close,
        servo_speed=record_cfg.servo_speed,
        servo_accel=record_cfg.servo_accel,
        servo_lookahead_time=record_cfg.servo_lookahead_time,
        servo_gain=record_cfg.servo_gain,
        control_period_s=1.0 / record_cfg.fps,
        cameras=camera_config,
    )

    robot = UR5e(robot_config)
    teleop = UR5eTeleop(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features,
                                          "observation",
                                          use_video=True)
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

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors())

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
            robot.send_action(
                build_joint_action(record_cfg.teleop_init_joint),
                move_slow=True,
            )

            input("请在确认机械臂移动到初始位姿后，调整遥操臂基本匹配机械臂的姿势，然后按 Enter 进行校准...\n")
            teleop.device.reset_smoothing()
            teleop.calibrate_delta()

            print("\n校准完毕。\n")

            if record_cfg.close_gripper_before_recording:
                if record_cfg.enable_gripper:
                    input("你已启动录制前关闭夹爪，请按 Enter 闭合夹爪...\n")
                    robot._gripper.set_pos(0)
                else:
                    logging.warning(
                        "close_gripper_before_recording is enabled, but gripper control is disabled."
                    )

            input("按 Enter 开始录制...\n")
            print("请使用遥操臂进行遥操作...\n")

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
