from abc import ABC
import numpy as np
import open3d.geometry as o3d
from utils.math_utils import rotation_z, get_transformation_matrix


class LabelObject(ABC):

    def __init__(self, pos_x, pos_y, pos_z, length, width, height, rotation, label):
        self._rotation_z: float = rotation
        self._label: str = label
        self._bbox: o3d.OrientedBoundingBox = o3d.OrientedBoundingBox(np.array([pos_x, pos_y, pos_z]),
                                                                      rotation_z(rotation),
                                                                      np.array([length, width, height]))

    def transform(self, transformation_matrix: np.ndarray) -> None:
        self._bbox.transform(transformation_matrix)

    def translate(self, translation_matrix: np.ndarray, relative=True) -> None:
        self._bbox.translate(translation_matrix, relative=relative)

    def rotate(self, rotation_matrix: np.ndarray, center=(0, 0, 0)) -> None:
        self._bbox.rotate(rotation_matrix, center=center)

    def get_bounding_box(self):
        return self._bbox

    def get_label(self):
        return self._label
    def __str__(self):
        return f"class: {self.label_class}\t" \
               f"location: [{self.x}, {self.y}, {self.z}]\t" \
               f"dimension: [{self.length}, {self.width}, {self.height}]\t" \
               f"rotation: {self.rotation}"

    def __repr__(self):
        return self.__str__()
