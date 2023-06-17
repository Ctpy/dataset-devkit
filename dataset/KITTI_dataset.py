from pathlib import Path
import numpy as np
from point_cloud_object import PointCloudObject
from dataset import Dataset
from label.label_object import LabelObject
from typing import Union


class KITTIDataset(Dataset):

    def __init__(self, dataset_path: str):
        """
        KITTI dataset object
        :param dataset_path (str): Path to dataset folder structure
        """
        super().__init__(dataset_path)
        self.set = 'training'
        self.point_cloud_path = self._dataset_path / self.set / 'velodyne'
        self.label_path = self._dataset_path / self.set / 'label_2'
        self.calib_path = self._dataset_path / self.set / 'calib'
        self.calib_files: list[Path] = []
        self.point_cloud_files.extend(self.point_cloud_path.glob('*.bin'))
        self.calib_files.extend(self.calib_path.glob('*.txt'))
        self.label_files.extend(self.label_path.glob('*.txt'))

    def load_frame(self, index: Union[int, Path]) -> np.ndarray:
        """
        Loads a frame from the given file where the dimension in KITTI is 4 [x, y, z, intensity]
        :param: index (Union[int, Path]) to point_cloud_file list or path to point cloud file
        :return: point cloud (np.ndarray)
        """

        if isinstance(index, int):
            point_cloud_file: Path = self.point_cloud_files[index]
            label_file: Path = self.label_files[index]
            calib_file: Path = self.calib_files[index]
        elif isinstance(index, str):
            point_cloud_file: Path = Path(index)
            label_file: Path = Path(index.replace('velodyne', 'label_2').replace('bin', 'txt'))
            calib_file: Path = Path(index.replace('velodyne', 'calib').replace('bin', 'txt'))
        else:
            raise TypeError("Expected int or str got %s", type(index))

        with open(point_cloud_file.resolve(), 'rb') as f:
            data = f.read()
            assert len(data) % 4 == 0
            point_cloud = np.frombuffer(data, dtype=np.float32).reshape((-1, 4))
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
        return point_cloud

    @staticmethod
    def __read_label_file(file_path: Path) -> list[LabelObject]:
        with open(file_path.resolve(), 'r') as f:
            data = f.read()

        label_object_list: list[LabelObject] = []

        for line in data.split('\n')[:-1]:
            properties = line.split(' ')

            if properties[0] == 'DontCare':
                continue

            label_class = properties[0]
            height, width, length, pos_x, pos_y, pos_z, rotation = tuple(prop for prop in properties[8:15])
            label_object_list.append(LabelObject(pos_x, pos_y, pos_z, length, width, height, rotation, label_class))

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
            elif line.split(' ')[0] == "Tr_velo_to_cam:":
                transform_matrix = np.loadtxt(line.split(' ')[1:]).reshape((3, 4))

        if rotation_matrix is None or transform_matrix is None:
            raise ValueError("Rotation or translation matrix are not found in file %s", file_path.resolve())

        return rotation_matrix, transform_matrix


if __name__ == '__main__':
    kitti = KITTIDataset("D:\\Datasets\\KITTI")
    print(kitti.load_frame(1).shape)

