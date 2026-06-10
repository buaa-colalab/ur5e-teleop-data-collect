import cv2
import numpy as np
import rtde_receive
import rtde_control
import time
from robotiq_gripper import RobotiqGripper
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# ===== Configuration =====
ROBOT_IP = "192.168.119.3"  # 修改为你的UR5e机械臂IP地址
GRIPPER_PORT = 63352
GRIPPER_FORCE = 255
VIDEO0_ID = 0  # main camera
VIDEO2_ID = 2  # wrist camera
PROMPT = "Pick up the measuring cylinder and pour an appropriate amount of water into the beaker."
GRIPPER_POSITION_RAW_MIN = 0.0
GRIPPER_POSITION_RAW_MAX = 255.0
GRIPPER_POSITION_NORM_MIN = 0.0
GRIPPER_POSITION_NORM_MAX = 1.0
EXEC_STEPS = 15  # 每次推理执行的步数（总预测步数为50）
SKIP_STEPS = 2  # 每次推理跳过的初始步数（如果模型前几步预测不稳定，可以设置为1或2）

# ===== Initialize policy =====
print("[INFO] Loading policy...")
config = _config.get_config("pi05_ur5e_full")
checkpoint_dir = "/home/lab/openpi/checkpoints/pi05_ur5e_full/ur5e_pour_water/29999"
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print("[INFO] Policy loaded successfully!")

# ===== Initialize cameras =====
print("[INFO] Initializing cameras...")
cap_main = cv2.VideoCapture(VIDEO0_ID)
cap_wrist = cv2.VideoCapture(VIDEO2_ID)

if not cap_main.isOpened() or not cap_wrist.isOpened():
    print("[ERROR] Failed to open camera(s)!")
    exit(1)

# 设置摄像头分辨率（可选，根据需要调整）
# cap_main.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap_main.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Cameras initialized!")

# ===== Initialize UR5e connection =====
print(f"[INFO] Connecting to UR5e at {ROBOT_IP}...")
try:
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    print("[INFO] Connected to UR5e successfully!")
except Exception as e:
    print(f"[ERROR] Failed to connect to UR5e: {e}")
    exit(1)

rtde_c.moveJ([-1.6285174528705042, -1.6196133098998011, 2.112854782735006, -0.43745441854510503, 1.5982029438018799, 0], speed=0.5, acceleration=0.5)

print(f"[INFO] Connecting to Robotiq gripper at {ROBOT_IP}:{GRIPPER_PORT}...")
try:
    gripper = RobotiqGripper()
    gripper.connect(ROBOT_IP, GRIPPER_PORT)
    gripper.activate(auto_calibrate=True)
    print("[INFO] Robotiq gripper connected successfully!")
except Exception as e:
    print(f"[ERROR] Failed to connect to Robotiq gripper: {e}")
    exit(1)


def _read_state():
    joint_positions = rtde_r.getActualQ()
    gripper_position_raw = float(gripper.get_current_position())
    gripper_position_norm = 1 - normalize_gripper_position(gripper_position_raw)
    state = np.asarray(list(joint_positions) + [gripper_position_norm], dtype=np.float32)
    return state, gripper_position_raw, gripper_position_norm


def normalize_gripper_position(
    gripper_position,
    raw_min=GRIPPER_POSITION_RAW_MIN,
    raw_max=GRIPPER_POSITION_RAW_MAX,
    norm_min=GRIPPER_POSITION_NORM_MIN,
    norm_max=GRIPPER_POSITION_NORM_MAX,
):
    if raw_max == raw_min:
        raise ValueError("raw_min and raw_max must be different")

    clipped_position = float(np.clip(gripper_position, raw_min, raw_max))
    ratio = (clipped_position - raw_min) / (raw_max - raw_min)
    return norm_min + ratio * (norm_max - norm_min)


def denormalize_gripper_position(
    gripper_position,
    raw_min=GRIPPER_POSITION_RAW_MIN,
    raw_max=GRIPPER_POSITION_RAW_MAX,
    norm_min=GRIPPER_POSITION_NORM_MIN,
    norm_max=GRIPPER_POSITION_NORM_MAX,
):
    if norm_max == norm_min:
        raise ValueError("norm_min and norm_max must be different")

    clipped_position = float(np.clip(gripper_position, norm_min, norm_max))
    ratio = (clipped_position - norm_min) / (norm_max - norm_min)
    return raw_min + ratio * (raw_max - raw_min)


def _overlay_info(frame, gripper_position_raw, gripper_position_norm):
    cv2.putText(
        frame,
        f"gripper: {gripper_position_raw:.1f} -> {gripper_position_norm:.3f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

# ===== Main inference loop =====
print("[INFO] Starting inference loop. Press 'q' to quit, 's' to save frames...")
frame_count = 0

try:
    while True:
        # ===== Read cameras =====
        ret_main, frame_main = cap_main.read()
        ret_wrist, frame_wrist = cap_wrist.read()
        
        if not ret_main or not ret_wrist:
            print("[WARNING] Failed to read from camera(s)")
            continue
        
        # ===== Add text labels to frames =====
        cv2.putText(frame_main, "main", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        cv2.putText(frame_wrist, "wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)

        try:
            state, gripper_position_raw, gripper_position_norm = _read_state()
        except Exception as e:
            print(f"[WARNING] Failed to read UR5e state: {e}")
            continue

        _overlay_info(frame_main, gripper_position_raw, gripper_position_norm)
        _overlay_info(frame_wrist, gripper_position_raw, gripper_position_norm)

        # ===== Display frames =====
        cv2.imshow("Main Camera", frame_main)
        cv2.imshow("Wrist Camera", frame_wrist)
        
        # ===== Keyboard control (check quickly before inference) =====
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting...")
            break
        elif key == ord('s'):
            # Save current frames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"main_{timestamp}.jpg", frame_main)
            cv2.imwrite(f"wrist_{timestamp}.jpg", frame_wrist)
            print(f"[INFO] Frames saved: main_{timestamp}.jpg, wrist_{timestamp}.jpg")
        
        # ===== Prepare inference input =====
        # Preprocess images if needed (resize, normalize, etc.)
        # This depends on your policy's expected image format
        image_main = cv2.cvtColor(frame_main, cv2.COLOR_BGR2RGB)
        image_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
        
        example = {
            "observation/image": image_main,
            "observation/wrist_image": image_wrist,
            "observation/state": state,
            "prompt": PROMPT
        }
        
        # ===== Run inference =====
        try:
            action_chunk = np.asarray(policy.infer(example)["actions"])
            action_chunk = action_chunk[SKIP_STEPS:]  # 跳过前 SKIP_STEPS 步，执行接下来的 EXEC_STEPS 步
            print(f"[FRAME {frame_count}] Action chunk shape: {action_chunk.shape} Action chunk: {action_chunk}")
            
            # ===== Execute first EXEC_STEPS actions from the chunk =====
            # UR5e 前六维是机械臂绝对位置，第七维是 Robotiq 夹爪动作（归一化到 [0, 1]）
            for step_idx in range(min(EXEC_STEPS, len(action_chunk))):
                action = action_chunk[step_idx]
                
                arm_action = action[:6]
                gripper_action = float(action[6]) if action.shape[0] > 6 else 0.0
                gripper_action = float(np.clip(gripper_action, 0.0, 1.0))
                gripper_target = int(round(denormalize_gripper_position(1 - gripper_action)))

                print(f"[DEBUG] Step {step_idx}: arm_action={arm_action}, gripper_action={gripper_action:.3f}, gripper_target={gripper_target}")

                # exit(0)
                
                try:
                    # 执行机械臂绝对位置移动
                    rtde_c.moveJ(arm_action, speed=0.5, acceleration=0.5)
                    # 执行夹爪动作
                    gripper.move(gripper_target, speed=255, force=GRIPPER_FORCE)
                    print(f"[EXEC {frame_count}-{step_idx}] Arm: {arm_action}, Gripper: {gripper_target}")
                except Exception as e:
                    print(f"[ERROR] Action execution failed at step {step_idx}: {e}")
                    break
            
            # break
            
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            continue
        
        frame_count += 1

except KeyboardInterrupt:
    print("[INFO] Interrupted by user")

finally:
    # ===== Cleanup =====
    print("[INFO] Cleaning up...")
    cap_main.release()
    cap_wrist.release()
    cv2.destroyAllWindows()
    gripper.disconnect()
    rtde_r.disconnect()
    print("[INFO] Done!")