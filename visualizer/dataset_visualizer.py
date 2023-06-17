import numpy as np

from dataset.dataset import Dataset
from dataset.KITTI_dataset import KITTIDataset
import open3d as o3d


class DatasetVisualizer:

    def __init__(self, dataset: Dataset):
        self.dataset: Dataset = dataset

    def visualize(self) -> None:
        """
        Visualizes the whole point cloud frame with all labelled objects
        :return: None
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.dataset.get_point_cloud()[:, :3])
        label_objects = self.dataset.get_objects()
        geometries = [point_cloud]
        for label in label_objects:
            label.get_bounding_box().color = self.dataset.label_colors[self.dataset.label_mapping(label.get_label())]
            geometries.append(label.get_bounding_box())
        o3d.visualization.draw_geometries(geometries)

    def visualize_objects(self) -> None:
        """
        Visualizes every object of the loaded point cloud frame
        :return: None
        """
        if not self.dataset.get_frame_loaded():
            raise RuntimeError("Load frame before visualizing objects")
        point_cloud_objects = self.dataset.crop_objects()
        for point_cloud_object in point_cloud_objects:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(point_cloud_object.get_points()[:, :3])
            bounding_box = point_cloud_object.get_bounding_box()
            bounding_box.color = self.dataset.label_colors[self.dataset.label_mapping(point_cloud_object.get_label())]
            o3d.visualization.draw_geometries([point_cloud, bounding_box])


if __name__ == '__main__':
    kitti = KITTIDataset("D:\\Datasets\\KITTI")
    kitti.load_frame(3)
    print(kitti.get_objects())
    dataset_visualizer = DatasetVisualizer(kitti)
    dataset_visualizer.visualize_objects()
