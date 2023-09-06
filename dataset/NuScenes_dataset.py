from pathlib import Path
import numpy as np
from dataset import Dataset, LabelObject, PointCloudObject
import pickle

replace_str = "/mnt/wwn-0x5000c500ecc16c3d-part1/Nuscenes"


class NuScenesDataset(Dataset):

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, 10)
        self.set = 'training'
        self.point_cloud_path_sweep = self._dataset_path / 'sweeps' / 'LIDAR_TOP'
        self.point_cloud_path_sample = self._dataset_path / 'samples' / 'LIDAR_TOP'
        self.train_pickle_file_path = self._dataset_path / 'nuscenes_infos_train.pkl'
        self.val_pickle_file_path = self._dataset_path / 'nuscenes_infos_val.pkl'
        self.train_pickle = self.load_pickle_file('train')
        self.val_pickle = self.load_pickle_file('val')
        self.point_cloud_files.extend(self.point_cloud_path_sample.glob('*.bin'))
        self.label_map = {'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3, 'construction_vehicle': 4, 'pedestrian': 5,
                          'motorcycle': 6, 'bicycle': 7, 'traffic_cone': 8, 'barrier': 9}
        self.label_colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [0.5, 0.5, 0],  # Olive
            [0.5, 0, 0.5],  # Purple
            [0, 0.5, 0.5],  # Teal
            [0.5, 0.5, 0.5]  # Gray
        ]

    def load_frame(self, index: [int, Path]) -> None:
        frame = self.train_pickle['infos'][index]
        suffix_path = (frame['lidar_path'].replace(replace_str, ''))
        point_cloud_path = frame['lidar_path'].replace(replace_str, self._dataset_path.as_posix())
        num_features = frame['num_features']
        with open(point_cloud_path, 'rb') as f:
            data = f.read()
            assert len(data) % num_features == 0, f"Expected point cloud to have {num_features} features got {len(data) % num_features}"
        point_cloud = np.frombuffer(data, dtype=np.float32).reshape((-1, num_features))

        object_list: list[PointCloudObject] = []
        for i in range(len(frame['gt_boxes'])):
            if frame['valid_flag'][i]:
                x, y, z, length, width, height, yaw = tuple(prop for prop in frame['gt_boxes'][i])
                label_object = LabelObject(x, y, z, length, width, height, yaw, frame['gt_names'][i])
                object_list.append(PointCloudObject(label_object))

        self.set_loaded(True)
        self.set_point_cloud(point_cloud)
        self.set_labels(object_list)

    def load_pickle_file(self, split: str) -> list:
        if split == 'train':
            path = self.train_pickle_file_path
        elif split == 'val':
            path = self.val_pickle_file_path
        else:
            raise ValueError("Expected split to be 'train' or 'val' got %s", split)
        with open(path, 'rb') as f:
            pickle_file = pickle.load(f)
        return pickle_file

    def load_point_cloud(self, index: int) -> np.ndarray:
        pass


if __name__ == '__main__':
    nuscenes = NuScenesDataset("F:\\Nuscenes")
    print(nuscenes.train_pickle['infos'][100])