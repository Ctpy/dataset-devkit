import numpy as np
import open3d.geometry as o3d


class PointCloudObject:

    def __init__(self, points: np.ndarray, bbox: o3d.OrientedBoundingBox, label: str = None):
        self.__points: np.ndarray = points
        self.__bbox: o3d.OrientedBoundingBox = bbox
        self.__label: str = label

    def update(self, points: np.ndarray, bbox: o3d.OrientedBoundingBox) -> bool:
        assert self.__points.shape == points.shape
        # Sanity check if all points are inside bbox
        return True

    def rotate(self, rotation_matrix: np.ndarray) -> None:
        pass

    def translate(self, translation_matrix: np.ndarray) -> None:
        pass

    def transform(self, translation_matrix: np.ndarray = None, rotation_matrix: np.ndarray = None,
                  transformation: np.ndarray = None) -> None:
        pass

    def get_points(self) -> np.ndarray:
        return self.__points

    def get_bounding_box(self) -> o3d.OrientedBoundingBox:
        return self.__bbox

    def get_label(self) -> str:
        return self.__label
