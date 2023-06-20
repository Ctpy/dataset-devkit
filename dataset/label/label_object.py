from abc import ABC
import numpy as np
import open3d.geometry as o3d
from utils.math_utils import *


class LabelObject(ABC):

    def __init__(self, pos_x: float, pos_y: float, pos_z: float, length: float, width: float, height: float,
                 rotation: float, label: str):
        self._rotation_z: float = rotation
        self._label: str = label
        self._bbox: o3d.OrientedBoundingBox = o3d.OrientedBoundingBox(np.array([pos_x, pos_y, pos_z]),
                                                                      rotation_y(rotation),
                                                                      np.array([length, width, height]))

    def transform(self, transformation_matrix: np.ndarray) -> None:
        self._bbox.transform(transformation_matrix)

    def translate(self, translation_matrix: np.ndarray, relative=True) -> None:
        self._bbox.translate(translation_matrix, relative=relative)

    def rotate(self, rotation_matrix: np.ndarray, center=(0, 0, 0)) -> None:
        self._bbox.rotate(rotation_matrix, center=center)

    def get_bounding_box(self) -> o3d.OrientedBoundingBox:
        return self._bbox

    def get_rotation(self) -> float:
        return self._rotation_z

    def get_label(self) -> str:
        return self._label

    def __str__(self):
        return f"class: {self._label}\t" \
               f"location: [{self._bbox.center}]\t" \
               f"dimension: {self._bbox.extent}\t" \
               f"rotation: {self._rotation_z}"

    def __repr__(self):
        return self.__str__()
