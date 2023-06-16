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
        self.point_cloud_files.extend(self.point_cloud_path.glob('*.bin'))
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
        elif isinstance(index, str):
            point_cloud_file: Path = Path(index)
            label_file: Path = Path(index.replace('velodyne', 'label_2').replace('bin', 'txt'))
        else:
            raise TypeError("Expected int or str got %s", type(index))

        with open(point_cloud_file.resolve(), 'rb') as f:
            data = f.read()
            assert len(data) % 4 == 0
            point_cloud = np.frombuffer(data, dtype=np.float32).reshape((-1, 4))
        self.__read_label_file(label_file)
        return point_cloud

    @staticmethod
    def __read_label_file(file_path: Path) -> list[LabelObject]:
        with open(file_path.resolve(), 'r') as f:
            data = f.read()

        label_object_list: list[LabelObject] = []

        for line in data.split('\n')[:-1]:
            properties = line.split(' ')
            print(properties)

        return label_object_list

    def get_objects(self) -> list[PointCloudObject]:
        pass

    def crop_object(self, index: int) -> PointCloudObject:
        pass

    def crop_objects(self) -> list[PointCloudObject]:
        pass


if __name__ == '__main__':
    kitti = KITTIDataset("D:\\Datasets\\KITTI")
    print(kitti.load_frame(1).shape)
