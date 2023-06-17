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
        self._loaded_labels: list[PointCloudObject] = []

    def _set_loaded(self, is_loaded: bool) -> None:
        """
        Set status if the dataset object has loaded a frame
        :param is_loaded: (bool) dataset load frame
        :return: None
        """
        assert self._loaded_labels or self._loaded_point_cloud
        self._frame_loaded = is_loaded

    def _set_point_cloud(self, point_cloud: np.ndarray) -> bool:
        # TODO: Add checks
        self._loaded_point_cloud = point_cloud
        return True

    def _set_labels(self, labels: list[PointCloudObject]) -> bool:
        # TODO: Add checks
        self._loaded_labels = labels
        return True

    def clear_frame(self) -> None:
        """
        Clears dataset from previously loaded frame and labels
        :return: None
        """
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

    def get_objects(self) -> list[PointCloudObject]:
        """
        Extract the labels
        :return: PointCloudObject
        """
        assert self._frame_loaded, "Load frame before accessing PointCloudLabel objects"
        return self._loaded_labels

    def crop_object(self, index: int) -> PointCloudObject:
        """
        Crops the point cloud to the bounding box
        :param index: (int) Index of the label object in loaded label list
        :return: (PointCloudObject)
        """
        self._loaded_labels[index].crop(self._loaded_point_cloud)
        return self._loaded_labels[index]

    def crop_objects(self) -> list[PointCloudObject]:
        """
        Crops point cloud to all bounding boxes in loaded label list
        :return: (List[PointCloudObject])
        """
        for label in self._loaded_labels:
            label.crop(self._loaded_point_cloud)
        return self._loaded_labels
