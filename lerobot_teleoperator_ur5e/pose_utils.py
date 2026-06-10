"""Utility functions for pose transformations and formatting."""

import numpy as np


def format_python_list(values: np.ndarray) -> str:
    """Format numpy array as a Python list string."""
    return np.array2string(
        values,
        precision=4,
        separator=", ",
        max_line_width=100000,
    )


def skew(vector: np.ndarray) -> np.ndarray:
    """Convert a 3D vector to its skew-symmetric matrix."""
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ],
        dtype=float,
    )


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to rotation matrix using Rodriguez formula."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)

    axis = rotvec / theta
    skew_axis = skew(axis)
    return (
        np.eye(3)
        + np.sin(theta) * skew_axis
        + (1.0 - np.cos(theta)) * (skew_axis @ skew_axis)
    )


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to roll-pitch-yaw angles."""
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    cos_pitch = float(np.cos(pitch))

    if abs(cos_pitch) < 1e-8:
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        yaw = 0.0
    else:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))

    return np.array([roll, pitch, yaw], dtype=float)


def rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to roll-pitch-yaw angles."""
    return matrix_to_rpy(rotvec_to_matrix(rotvec))


def axis_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis and angle using Rodriguez formula."""
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)

    unit_axis = axis / axis_norm
    skew_axis = skew(unit_axis)
    return (
        np.eye(3)
        + np.sin(angle) * skew_axis
        + (1.0 - np.cos(angle)) * (skew_axis @ skew_axis)
    )


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi] range."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def tcp_pose_to_joints_and_tcp(
    tcp_pose: np.ndarray,
    robot_joint_positions: np.ndarray,
    joint_coef: np.ndarray,
    teleoperator_joint_positions: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute target joint positions and tcp pose for action.
    
    Args:
        tcp_pose: Current TCP pose (6D: x, y, z, rx, ry, rz)
        robot_joint_positions: Current robot joint positions (6D)
        joint_coef: Joint coefficient scaling
        teleoperator_joint_positions: Current teleoperator joint positions (6D)
        delta: Calibration delta offset
        
    Returns:
        Tuple of (target_joint_positions, target_tcp_pose)
    """
    # Compute target joint from teleoperator with sync
    target_joint = joint_coef * teleoperator_joint_positions + delta
    
    # TCP pose is computed based on keyboard input or master arm position
    # This function returns both for consistency
    return target_joint, tcp_pose
