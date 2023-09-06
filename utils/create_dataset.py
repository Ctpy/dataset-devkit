import threading
from pathlib import Path

from dataset.NuScenes_dataset import NuScenesDataset
from dataset.dataset import Dataset
from dataset.KITTI_dataset import KITTIDataset
from dataset.point_cloud_object import PointCloudObject
from copy import deepcopy
from tqdm import tqdm
import argparse
import os
import json
import numpy as np
import open3d as o3d

class DatasetCreator:

    def __init__(self, dataset: str, input_path: str, output_path: str, num_workers: int = 1):
        self.dataset: Dataset = self.init_dataset(dataset, input_path)
        self.output_path: Path = Path(output_path)
        self.num_workers = num_workers
        self.create_directory()

    def init_dataset(self, dataset: str, input_path: str) -> Dataset:
        if dataset == 'kitti':
            return KITTIDataset(input_path)
        elif dataset == 'nuscenes':
            return NuScenesDataset(input_path)
        else:
            raise NotImplementedError

    def create_directory(self) -> None:
        label_path = Path(self.output_path) / 'labels'
        point_cloud_path = Path(self.output_path) / 'point_clouds'

        if not os.path.exists(label_path):
            os.makedirs(label_path)

        if not os.path.exists(point_cloud_path):
            os.makedirs(point_cloud_path)

    def create_dataset(self) -> None:
        bins = np.array_split(range(len(self.dataset)), self.num_workers)
        for idx in range(self.num_workers):
            thread = threading.Thread(target=self.convert, args=(idx, bins[idx],))
            thread.start()

    def convert(self, thread_id: int, bin: np.ndarray) -> None:
        dataset = deepcopy(self.dataset)
        for idx in tqdm(bin, desc=f"Thread {thread_id + 1}"):
            dataset.load_frame(int(idx))
            for i, point_cloud_object in enumerate(dataset.get_objects()):
                point_cloud_object.crop(dataset.get_point_cloud())
                point_cloud_object.normalize()
                self.export(point_cloud_object, f"{str(idx).zfill(4)}_{str(i).zfill(2)}")

    def export(self, point_cloud_object: PointCloudObject, filename: str) -> None:
        with open((self.output_path / 'labels' / (filename + '.json')).resolve(), 'w') as f:
            label_object = point_cloud_object.get_label_object()
            label = {
                'label': label_object.get_label(),
                'extent': list(label_object.get_bounding_box().extent),
                'location': list(label_object.get_bounding_box().center),
                'rotation': np.arctan2(label_object.get_bounding_box().R[2, 2],
                                       label_object.get_bounding_box().R[1, 1]),
                'num_points': int(point_cloud_object.get_points().shape[0])
            }
            json.dump(label, f)
            f.close()
        # np.save(str(self.output_path / 'point_clouds' / (filename + '.bin')), point_cloud_object.get_points())
        point_cloud_object.get_points().tofile(str(self.output_path / 'point_clouds' / (filename + '.bin')))

        # pcd = o3d.geometry.PointCloud()
        # points = point_cloud_object.get_points()
        # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # o3d.io.write_point_cloud(str(self.output_path / 'point_clouds' / (filename + '.pcd')), pcd)


def parse_arguments() -> tuple[str, str, str, int]:
    parser = argparse.ArgumentParser("Parsing arguments to init DatasetCreator")
    parser.add_argument('dataset',
                        type=str,
                        choices=["kitti", "nuscenes"],
                        help="Dataset you want to extract from and create")
    parser.add_argument('--input_path', '-i', required=True, type=str, help="Path to dataset")
    parser.add_argument('--output_path', '-o', required=True, type=str, help="Path to dataset")
    parser.add_argument('--worker', '-w', type=int, default=1, help="Number of threads used for dataset creation")
    args = parser.parse_args()
    print(args)
    return args.dataset, args.input_path, args.output_path, args.worker


if __name__ == '__main__':
    dataset_creator = DatasetCreator(*parse_arguments())
    dataset_creator.create_dataset()
