import sys, time
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any
from lerobot_robot_ur5e import UR5eConfig, UR5e
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


class ReplayConfig:
    def __init__(self, cfg: Dict[str, Any]):
        robot = cfg["robot"]

        # global config
        self.repo_id: str = cfg["repo_id"]
        self.episode_idx: str = cfg.get("episode_idx", 0)

        # robot config
        self.robot_ip: str = robot["ip"]
        self.enable_gripper: bool = bool(robot.get("enable_gripper", True))
        self.gripper_port: int = int(robot.get("gripper_port", 63352))
        self.gripper_open: int = int(robot.get("gripper_open", 0))
        self.gripper_close: int = int(robot.get("gripper_close", 255))
        self.gripper_force: int = int(robot.get("gripper_force", 255))
        self.gripper_init_open: bool = bool(
            robot.get("gripper_init_open", False)
        )
        self.gripper: dict[str, Any] = robot.get("gripper", {
            "choice": "robotiq",
            "robotiq": {
                "ip": self.robot_ip,
                "port": self.gripper_port,
                "force": self.gripper_force,
                "open": self.gripper_open,
                "close": self.gripper_close,
            },
        })


def main(replay_cfg: ReplayConfig):
    episode_idx = replay_cfg.episode_idx

    robot_config = UR5eConfig(
        robot_ip=replay_cfg.robot_ip,
        enable_gripper=replay_cfg.enable_gripper,
        gripper=replay_cfg.gripper,
        gripper_init_open=replay_cfg.gripper_init_open,
    )

    robot = robot = UR5e(robot_config)
    robot.connect()

    init_joint = [-1.6285174528705042, -1.6196133098998011, 2.112854782735006, -0.43745441854510503, 1.5982029438018799, 0]
    robot._arm["rtde_c"].moveJ(init_joint, speed = 0.2, acceleration = 1)
    robot.send_action({"gripper_position": 0.0 if replay_cfg.gripper_init_open else 1.0})
    time.sleep(3)

    dataset = LeRobotDataset(replay_cfg.repo_id, episodes=[episode_idx])
    actions = dataset.hf_dataset.select_columns("action")
    log_say(f"Replaying episode {episode_idx}")
    for idx in range(dataset.num_frames):
        t0 = time.perf_counter()

        action = {
            name: float(actions[idx]["action"][i])
            for i, name in enumerate(dataset.features["action"]["names"])
        }
        robot.send_action(action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

    robot.disconnect()


if __name__ == "__main__":
    with open(Path(__file__).parent / "config" / "cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    replay_cfg = ReplayConfig(cfg["replay"])

    main(replay_cfg)
