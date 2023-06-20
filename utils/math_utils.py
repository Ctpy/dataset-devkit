import numpy as np


def rotation_x(rotation: float) -> np.ndarray:
    rotation_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rotation), -np.sin(rotation)],
        [0.0, np.sin(rotation), np.cos(rotation)]])
    return rotation_matrix


def rotation_y(rotation: float) -> np.ndarray:
    rotation_matrix = np.array([
        [np.cos(rotation), 0.0, np.sin(rotation)],
        [0.0, 1, 0.0],
        [-np.sin(rotation), 0.0, np.cos(rotation)]])
    return rotation_matrix


def rotation_z(rotation: float) -> np.ndarray:
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])
    return rotation_matrix


def get_transformation_matrix(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # TODO: Add checks
    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Assign the translation vector to the last column of the transformation matrix
    transformation_matrix[:3, 3] = translation

    # Assign the rotation matrix to the upper-left 3x3 block of the transformation matrix
    transformation_matrix[:3, :3] = rotation

    return transformation_matrix
