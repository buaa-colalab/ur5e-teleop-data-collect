# UR5e&DH Data Collect Tutorial


## 开机操作
【step1】打开遥操臂底座箱里的插座开关
![](assets/0_main_power.png)
【step2】打开机械臂控制面板的开关(图里上方那个银色的开关按钮)，等待机械臂系统启动
![](assets/1_ur5e_after_power_on.png)
【step3】点击控制面板左下角的红色按钮，进入初始化面板
![](assets/2_ur5e_init_panel.png)
【step4】在弹出的界面中点开，等待
![](assets/3_ur5e_started.png)
【step5】在达到上图所示的进度后，再点一次开，等待如下图时，此时机械臂的所有关节锁已释放，可以开始控制机械臂的运动了
![](assets/4_ur5e_activated.png)


## 关机操作

【step1】点击控制面板左下角的绿色按钮，在弹出的界面中点关，等待机械臂上锁完成后退出
![](assets/4_ur5e_activated.png)
【step2】点击右上角的三根横线，再点击关闭机器人
![](assets/5_ur5e_menu.png)
【step3】关闭底座箱里的插座
![](assets/0_main_power.png)


## 通过控制面板控制机械臂运动

确保已按『开机操作』执行，初始化面板全绿，左下角显示绿色：正常

【step1】将右上角的控制模式切换为『本地控制』

![](assets/7.png)

【step2】点击左上角的『移动』进入本地控制，左上角为末端xyz，左下角为末端的rpy，右下角可直接控制每个关节的旋转角度

![](assets/8.png)


## 通过远程python代码控制机械臂运动与夹爪状态

确保已按『开机操作』执行，初始化面板全绿，左下角显示绿色：正常

【step1】将右上角的控制模式切换为『远程控制』

【step2】详见 `tutorial_basic_sdk_usage.ipynb`


## 使用遥操臂或键盘进行数据采集

确保已按『开机操作』执行，初始化面板全绿，左下角显示绿色：正常，激活 conda 环境 `conda activate ur5e_demo`

【step1】将右上角的控制模式切换为『远程控制』

【step2】按需更改 `scripts/config/cfg.yaml` 的关键参数：
- 数据集名称：`repo_id`，会存储在 `~/.cache/huggingface/lerobot/{repoid}` 中，如果要重新采集，需自行删除。如果要在已有数据集上进行扩增采集，将配置文件中的 `resume` 改为 `True`（未测试过 `resume=True` 的有效性，谨慎使用）
- 采集轨迹数据条数：`num_episodes`
- 任务指令：`description`
- 初始位姿： `teleop_init_joint` 或 `keyboard_init_joint`
- 每条数据记录前是否先闭合夹爪: `close_gripper_before_recording`

【step3】（如果使用遥操臂）打开遥操臂开关（图中绿色按键，亮起表示通电），遥操臂右侧按键（图中蓝色按钮）熄灭时可以自由移动，亮起时锁定，按下该键以切换这两种状态。遥操臂的扳机按一次会使夹爪在“完全开”和“完全闭”两种状态间进行切换（请在录制中按，其他时候按无效）。`standalone_teleop_ur5e_only.py`是只遥操（不包含夹爪）、不进行数据采集的代码。
![](assets/4_ur5e_activated.png)

【step4】运行遥操脚本：`python scripts/run_record_teleop.py` 或 `python scripts/run_record_keyboard.py`，脚本会打印操作说明，并给出流程引导，依照执行即可

## 数据格式转换（以转换为Pi0的训练格式为例）

scripts/convert.py可将所采数据转化为Pi0训练所需的格式。

```
python scripts/convert.py --input /home/user/project_0418/data/ur5e_transfer_wafer_part_2 --output-dir /home/user/project_0418/data/ur5e_transfer_wafer_part_2_v21
```
若要转换为其他格式，可自行参考 `scripts/convert.py` 编写。