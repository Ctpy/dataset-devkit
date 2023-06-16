from abc import ABC, abstractmethod
from pathlib import Path
from point_cloud_object import PointCloudObject
from typing import Union
import numpy as np


class Dataset(ABC):

    def __init__(self, dataset_path: str) -> None:
        self._dataset_path: Path = Path(dataset_path)
        self.point_cloud_path: Union[Path, None] = None
        self.label_path: Union[Path, None] = None
        self.point_cloud_files: list[Path] = []
        self.label_files: list[Path] = []
        self._frame_loaded: bool = False
        self._loaded_point_cloud: Union[np.ndarray, None] = None
        self._loaded_labels: Union[np.ndarray, None] = None

    def _set_loaded(self, is_loaded: bool) -> None:
        assert self._loaded_labels or self._loaded_point_cloud
        self._frame_loaded = is_loaded

    def clear_frame(self) -> None:
        self._frame_loaded = False
        self._loaded_point_cloud = None
        self._loaded_labels = None

    @abstractmethod
    def load_frame(self, index: [int, Path]) -> np.ndarray:
        """
        Loads a frame from dataset_path by indexing or with a Path object
        :param index: Integer or Path object to load from
        :return: np.ndarray
        """
        pass

    @abstractmethod
    def get_objects(self) -> list[PointCloudObject]:
        """
        Extract the labels
        :return: PointCloudObject
        """
        pass

    @abstractmethod
    def crop_object(self, index: int) -> PointCloudObject:
        pass

    @abstractmethod
    def crop_objects(self) -> list[PointCloudObject]:
        pass

    def _rotate(self, rotation_matrix: np.ndarray) -> None:
        pass

    def _translate(self, translation_matrix: np.ndarray) -> None:
        pass

    def _normalize(self) -> None:
        pass
