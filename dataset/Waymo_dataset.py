from pathlib import Path
import numpy as np
from dataset.point_cloud_object import PointCloudObject
from dataset import Dataset
from dataset.label.label_object import LabelObject
from typing import Union
from utils.math_utils import *

class WaymoDataset(Dataset):

    def __init__(self, dataset_path: str):
        """
        KITTI dataset object
        :param dataset_path (str): Path to dataset folder structure
        """
        super().__init__(dataset_path, 7)
        self.set = 'training'
        self.point_cloud_path = self._dataset_path / self.set / 'velodyne'
        self.label_path = self._dataset_path / self.set / 'label_all'
        self.calib_path = self._dataset_path / self.set / 'calib'
        self.calib_files: list[Path] = []
        self.point_cloud_files.extend(self.point_cloud_path.glob('*.bin'))
        self.calib_files.extend(self.calib_path.glob('*.txt'))
        self.label_files.extend(self.label_path.glob('*.txt'))

        # assert len(self.point_cloud_files) == len(
        #     self.label_files), f"Size mismatch between point cloud files " \
        #                        f"{len(self.point_cloud_files)} and labels {len(self.label_files)}"

        self.label_map = {'Car': 0, ' PEDESTRIAN': 1, 'CYCLIST': 2, 'SIGN': 3}
        self.label_colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0]  # Yellow
        ]

    def load_frame(self, index: Union[int, Path]) -> None:
        """
        Loads a frame from the given file where the dimension in KITTI is 4 [x, y, z, intensity]
        :param: index (Union[int, Path]) to point_cloud_file list or path to point cloud file
        :return: None
        """

        if isinstance(index,  int):
            point_cloud_file: Path = self.point_cloud_files[index]
            label_file: Path = self.label_files[index]
            calib_file: Path = self.calib_files[index]
        elif isinstance(index, str):
            point_cloud_file: Path = Path(index)
            label_file: Path = Path(index.replace('velodyne', 'label_2').replace('bin', 'txt'))
            calib_file: Path = Path(index.replace('velodyne', 'calib').replace('bin', 'txt'))
        else:
            raise TypeError("Expected int or str got %s", str(type(index)))

        with open(point_cloud_file.resolve(), 'rb') as f:
            data = f.read()
            assert len(data) % 6 == 0
            point_cloud = rotate_point_cloud(np.frombuffer(data, dtype=np.float32).reshape((-1, 6)), -np.pi / 3, axis='z')
        label_object_list = self.__read_label_file(label_file)
        rotation_matrix, transform_matrix = self.__read_calib_files(calib_file)

        rotation_matrix = np.linalg.inv(rotation_matrix)
        transform_matrix = np.linalg.inv(transform_matrix)

        # Perform rotation of labels
        for label_object in label_object_list:
            label_object.rotate(rotation_matrix)
            label_object.rotate(transform_matrix[:3, :3])
            label_object.translate(transform_matrix[:3, 3])

        point_cloud_object_list: list[PointCloudObject] = []
        for label_object in label_object_list:
            point_cloud_object_list.append(PointCloudObject(label_object))

        # Set all data
        self.set_loaded(True)
        self.set_point_cloud(point_cloud)
        self.set_labels(point_cloud_object_list)

    @staticmethod
    def __read_label_file(file_path: Path) -> list[LabelObject]:
        with open(file_path.resolve(), 'r') as f:
            data = f.read()

        label_object_list: list[LabelObject] = []

        for line in data.split('\n')[:-1]:
            properties = line.split(' ')

            if properties[0] in ['DontCare', 'Misc']:
                continue

            label_class = properties[0]
            height, width, length, pos_x, pos_y, pos_z, rotation = tuple(prop for prop in properties[8:15])
            label_object_list.append(
                LabelObject(float(pos_x), float(pos_y) - float(height) / 2, float(pos_z), float(length), float(height),
                            float(width), float(rotation), label_class, rotation_setting='y'))

        return label_object_list

    @staticmethod
    def __read_calib_files(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        rotation_matrix: Union[np.ndarray, None] = None
        transform_matrix: Union[np.ndarray, None] = None

        with open(file_path.resolve(), 'r') as f:
            data = f.read()

        for line in data.split('\n'):
            if line.split(' ')[0] == "R0_rect:":
                rotation_matrix = np.loadtxt(line.split(' ')[1:]).reshape((3, 3))
            elif line.split(' ')[0] == "Tr_velo_to_cam_2:":
                transform_matrix = np.loadtxt(line.split(' ')[1:]).reshape((3, 4))

        if rotation_matrix is None or transform_matrix is None:
            raise ValueError("Rotation or translation matrix are not found in file %s", file_path.resolve())

        return rotation_matrix, np.append(transform_matrix, np.array([[0, 0, 0, 1]]), axis=0)
