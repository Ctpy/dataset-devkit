import numpy as np
import open3d as o3d
from dataset.label.label_object import LabelObject
from utils.math_utils import *
from typing import Union, Optional


class PointCloudObject:

    def __init__(self, label_object: LabelObject):
        self.__points: Union[np.ndarray, None] = None
        self.__label: LabelObject = label_object

    def crop(self, points: np.ndarray) -> None:
        points_vector = o3d.utility.Vector3dVector(points[:, :3])
        point_indices = self.__label.get_bounding_box().get_point_indices_within_bounding_box(points_vector)
        self.__points = points[point_indices]

    def rotate(self, rotation: Union[np.ndarray, float], dim: Optional[int] = None) -> None:
        if dim is not None and isinstance(rotation, float):

            if dim == 0:
                rotation_matrix = rotation_x(rotation)
            elif dim == 1:
                rotation_matrix = rotation_y(rotation)
            elif dim == 2:
                rotation_matrix = rotation_z(rotation)
            else:
                raise ValueError("dim must be 0 <= dim < 3 got %s", dim)

        elif dim is None and isinstance(rotation, np.ndarray):
            assert rotation.shape[0] == rotation.shape[1], "Expected matrix to be squared"
            assert rotation.shape[0] == 3, f"Expected matrix of size 3 got {rotation.shape[0]}"
            rotation_matrix = rotation

        else:
            raise TypeError("Expected rotation of type np.ndarray got %s or float and dim of type int or None got %s",
                            type(rotation), type(dim))

        self.__points[:, :3] = self.__points[:, :3] @ rotation_matrix.T
        self.__label.rotate(rotation_matrix)

    def translate(self, translation_matrix: np.ndarray) -> None:
        pass

    def transform(self, translation_matrix: np.ndarray = None, rotation_matrix: np.ndarray = None,
                  transformation: np.ndarray = None) -> None:
        pass

    def normalize(self):
        center = np.asarray(self.__label.get_bounding_box().center)
        fill_length = self.__points.shape[1] - len(center)
        zero_matrix = np.zeros(fill_length)
        center = np.concatenate((center, zero_matrix))
        normalized_points = self.__points - center
        self.__points = normalized_points
        self.__label.get_bounding_box().center = np.array([0, 0, 0])

    def update(self, points: np.ndarray, bbox: o3d.geometry.OrientedBoundingBox) -> bool:
        assert self.__points.shape == points.shape
        # Sanity check if all points are inside bbox
        return True

    def get_points(self) -> np.ndarray:
        return self.__points

    def get_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        return self.__label.get_bounding_box()

    def get_label_object(self) -> LabelObject:
        return self.__label

    def __repr__(self):
        return self.__label.__repr__()
