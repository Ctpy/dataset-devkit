import numpy as np
import open3d as o3d
from label.label_object import LabelObject
from typing import Union


class PointCloudObject:

    def __init__(self, label_object: LabelObject):
        self.__points: Union[np.ndarray, None] = None
        self.__label: LabelObject = label_object

    def crop(self, points: np.ndarray) -> None:
        points_vector = o3d.utility.Vector3dVector(points)
        point_indices = self.__label.get_bounding_box().get_point_indices_within_bounding_box(points_vector)
        self.__points = points[point_indices]

    def rotate(self, rotation_matrix: np.ndarray) -> None:
        pass

    def translate(self, translation_matrix: np.ndarray) -> None:
        pass

    def transform(self, translation_matrix: np.ndarray = None, rotation_matrix: np.ndarray = None,
                  transformation: np.ndarray = None) -> None:
        pass

    def update(self, points: np.ndarray, bbox: o3d.geometry.OrientedBoundingBox) -> bool:
        assert self.__points.shape == points.shape
        # Sanity check if all points are inside bbox
        return True

    def get_points(self) -> np.ndarray:
        return self.__points

    def get_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        return self.__label.get_bounding_box()

    def get_label(self) -> str:
        return self.__label.get_label()
