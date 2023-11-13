import glob
import os
import json
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
VIS = False
ROOT_DIR = "C:\\Users\\Tung\\Desktop\\repos\\dataset-devkit"
# we read the filesnames from the txt file
filename_path = "C:\\Users\\Tung\\Desktop\\repos\\PointNeXt\\data\\CroppedKITTI\\test_1024.txt"
# inference_data_path = "C:\\Users\\Tung\\Desktop\\repos\\PointNeXt\\gaussian_inference_labels_3_clusters_2"
is_repsurf = True
mode = "RepSurf" if is_repsurf else ""
method = f"GaussianMixture"
folder = mode
if not is_repsurf:
    inference_data_path = "C:\\Users\\Tung\\Desktop\\repos\\PointNeXt\\predictions\\Spectral"
    npy_data_path = "D:\\Datasets\\KITTI_cropped\\point_clouds"
    test_pickle_file = "C:\\Users\\Tung\\Desktop\\repos\\PointNeXt\\data\\CroppedKITTI\\test_1024.pkl"
else:
    inference_data_path = f"C:\\Users\\Tung\\Desktop\\repos\\RepSurf\\predictions\\{method}"
    test_pickle_file = "C:\\Users\\Tung\\Desktop\\repos\\RepSurf\\predictions\\inference_pcd"
    npy_data_path = "D:\\Datasets\\KITTI_cropped\\point_clouds"
for filename in os.listdir(inference_data_path):
    # Check if filename contains an underscore
    if '_' in filename:
        # Construct the old and new file paths
        old_file_path = os.path.join(inference_data_path, filename)
        new_file_path = os.path.join(inference_data_path, filename.replace('_', ''))

        # Rename the file
        os.replace(old_file_path, new_file_path)
ground_truth_data = "C:\\Users\\Tung\\Desktop\\repos\\dataset-devkit\\data\\kitti_object_test-v0.1.json"
# test_pickle_file = "C:\\Users\\Tung\\Desktop\\repos\\PointNeXt\\data\\CroppedKITTI\\test_1024.pkl"
replace_dash = False
npy_data = glob.glob(os.path.join(npy_data_path, "*.npy"))
data_idx_mapping = {}
idx_data_mapping = {}
with open(filename_path, 'r') as f:
    data = f.read()
    data = data.split('\n')[:-1]
    print(data)

for i, filename in enumerate(data):
    data_idx_mapping[filename] = i
    idx_data_mapping[i] = filename

inference_data = glob.glob(os.path.join(inference_data_path, "*.pkl"))
print(inference_data)
inference_data_data_mapping = {}

for filename in data:
    inference_data_data_mapping[filename] = filename.replace('_', '')
if not is_repsurf:
    with open(test_pickle_file, 'rb') as f:
        test_data = pickle.load(f)
    print(test_data)
    # read gt file

    point_cloud_data = test_data['data']
    label_data = test_data['label']
    filename_data = test_data['filename']

with open(ground_truth_data, 'r') as f:
    gt_data = json.load(f)
    gt_data = gt_data['dataset']

categories = gt_data["task_attributes"]['categories']
print(categories)
label_mapping = {
    'background': 0
}
label_id = [0]
label_color = [
    [0, 0, 0],
    [255, 0, 0],  # red -> front
    [0, 255, 0],  # green ->
    [0, 0, 255],  # blue
    [255, 255, 0],  # yellow
    [0, 255, 255],  # cyan
]
for category in categories:
    label_mapping[category['name']] = category['id']
    label_id.append(category['id'])

label_color = np.asarray(label_color)

print(label_mapping)
print(label_id)
print(label_color)
gt_samples = gt_data['samples']


def calculate_iou(cluster_indices, label_indices):
    # Calculate intersection and union counts
    intersection = np.sum(cluster_indices & label_indices)
    union = np.sum(cluster_indices | label_indices)
    return intersection / union if union != 0 else 0


CLUSTER_LABEL_SIZE = 3

GT_LABEL_SIZE = 5
overall_iou = []
overall_iou2 = []
overall_accuracy = []
overall_f1_score = []
for gt_sample in tqdm(gt_samples):
    filename = gt_sample['name'][:-4]

    # Processing gt labels
    sample = gt_sample['labels']['ground-truth']
    if sample is None:
        continue
    label_status = sample['label_status']
    if label_status != 'LABELED':
        continue
    gt_annotations = sample['attributes']['annotations']
    print(gt_annotations)
    gt_annotations = sample['attributes']['annotations']
    gt_mapping = {0: 0}
    # point_cloud_gt = o3d.geometry.PointCloud()
    # point_cloud_gt.points = o3d.utility.Vector3dVector(point_cloud_data[i, :, :3].cpu().numpy())
    for gt_annotation in gt_annotations:
        gt_mapping[gt_annotation['id']] = gt_annotation['category_id']
    # create mapping from id to category
    gt_label_annotations = sample['attributes']['point_annotations']
    gt_annotations_mapped = []
    for gt_label_annotation in gt_label_annotations:
        gt_annotations_mapped.append(gt_mapping[gt_label_annotation])
    gt_annotations_mapped = np.asarray(gt_annotations_mapped)

    # Load full point cloud data
    if os.path.exists(ROOT_DIR + "\\data\\" + f"KITTI_test_sampled{mode}\\" + filename + ".npy"):
        try:
            npy_pcd = np.load(ROOT_DIR + "\\data\\" + f"KITTI_test_sampled{mode}\\" + filename + ".npy")
            gt_annotations_mapped = np.load(ROOT_DIR + "\\data\\" + f"KITTI_test_sampled_label{mode}\\" + filename + ".npy")

            inference_labels = pickle.load(
                open(os.path.join(inference_data_path, inference_data_data_mapping[filename] + '.pkl'), 'rb'))
            inference_labels = np.asarray(inference_labels) + 1
        except:
            continue
    else:
        try:
            print(npy_data_path, filename)
            npy_pcd = np.load(os.path.join(npy_data_path, filename + ".npy"))
        except:
            continue
        # Load inference data
        if not is_repsurf:
            file_idx = data_idx_mapping[filename]
            inference = point_cloud_data[file_idx, :, :3].cpu().numpy()
        else:
            try:
                inference = np.load(test_pickle_file + "\\" + filename.replace('_', '') + ".npy")
            except:
                continue
        print(npy_pcd.shape, inference.shape)

        # Matching indices
        matching_indices = []
        for j in range(inference.shape[0]):
            for k in range(npy_pcd.shape[0]):
                if np.isclose(inference[j], npy_pcd[k, :3], atol=1e-4).all():
                    matching_indices.append(k)
                    break

        assert len(matching_indices) == min(inference.shape[0], npy_pcd.shape[0]), f"Matching indices length {len(matching_indices)}"

        # Label processing of gt and prediction
        if not os.path.exists(ROOT_DIR + f"\\data\\" + f"KITTI_test_sampled{mode}"):
            os.makedirs(ROOT_DIR + f"\\data\\" + f"KITTI_test_sampled{mode}")

        npy_pcd_sampled = np.save(ROOT_DIR + f"\\data\\" + f"KITTI_test_sampled{mode}\\" + filename + ".npy",
                                  npy_pcd[matching_indices])
        npy_pcd = npy_pcd[matching_indices]

        gt_annotations_mapped = gt_annotations_mapped[matching_indices]
        gt_annotations = gt_annotations_mapped
        if not os.path.exists(ROOT_DIR + f"\\data\\" + f"KITTI_test_sampled_label{mode}"):
            os.makedirs(ROOT_DIR + f"\\data\\" + f"KITTI_test_sampled_label{mode}")
        np.save(ROOT_DIR + f"\\data\\" + f"KITTI_test_sampled_label{mode}\\" + filename + ".npy", gt_annotations)
        inference_labels = pickle.load(
            open(os.path.join(inference_data_path, inference_data_data_mapping[filename] + '.pkl'), 'rb'))
        inference_labels = np.asarray(inference_labels) + 1
    unique, counts = np.unique(inference_labels, return_counts=True)
    sorted_indices = np.argsort(-counts)

    # Get the top k clusters
    top_k = unique[sorted_indices[:3]]
    inference_labels = np.where(np.isin(inference_labels, top_k), inference_labels, 0)
    mapping = {cluster: i + 1 for i, cluster in enumerate(top_k)}
    inference_labels = np.array([mapping.get(x, 0) for x in inference_labels])
    # inference_color = label_color[inference_labels]
    # Create point cloud
    iou_matrix = np.zeros((CLUSTER_LABEL_SIZE, GT_LABEL_SIZE))
    adjusted_ground_truth_labels = np.where(gt_annotations_mapped == 0, -1, gt_annotations_mapped)
    filtered_unsupervised_labels = inference_labels[adjusted_ground_truth_labels != -1]
    filtered_point_cloud = npy_pcd[adjusted_ground_truth_labels != -1]
    adjusted_ground_truth_labels = adjusted_ground_truth_labels[adjusted_ground_truth_labels != -1]
    fig = plt.figure(figsize=(7, 7))
    ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    for label in np.unique(filtered_unsupervised_labels):
        ax2.scatter(filtered_point_cloud[filtered_unsupervised_labels == label, 0], filtered_point_cloud[filtered_unsupervised_labels == label, 1],
                    filtered_point_cloud[filtered_unsupervised_labels == label, 2], linestyle='None')
    ax2.axis('off')
    ax2.set_facecolor("none")
    fig.tight_layout()
    fig.patch.set_facecolor("none")
    ax2.grid(False)
    if os.path.exists(f'{folder}/{method}') == False:
        os.mkdir(f'{folder}/{method}')
    fig.savefig(f'{folder}/{method}/{filename}_clustering.pdf')
    fig.clf()
    if VIS:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])
        point_cloud_inferred = o3d.geometry.PointCloud()
        point_cloud_inferred.points = o3d.utility.Vector3dVector(inference)
        point_cloud_inferred.colors = o3d.utility.Vector3dVector(inference_color)
        vis.add_geometry(point_cloud_inferred)

        vis.run()
        screenshot = vis.capture_screen_float_buffer(False)
        output_image_path = 'C:\\Users\\Tung\\Desktop\\repos\\dataset-devkit\\visualization'
        o3d.io.write_image(os.path.join(output_image_path, filename + "_inference.png"), screenshot)
        vis.destroy_window()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])
        point_cloud_gt = o3d.geometry.PointCloud()
        point_cloud_gt.points = o3d.utility.Vector3dVector(npy_pcd[matching_indices, :3])
        point_cloud_gt.colors = o3d.utility.Vector3dVector(label_color[gt_annotations_mapped])
        vis.add_geometry(point_cloud_gt)

        vis.run()
        screenshot = vis.capture_screen_float_buffer(False)
        o3d.io.write_image(os.path.join(output_image_path, filename + "_gt_labelled.png"), screenshot)
        vis.destroy_window()

    for i in range(CLUSTER_LABEL_SIZE):
        for j in range(GT_LABEL_SIZE):
            cluster_indices = (filtered_unsupervised_labels - 1 == i)
            gt_label_indices = (adjusted_ground_truth_labels == j + 1)

            iou_matrix[i, j] = calculate_iou(cluster_indices, gt_label_indices)

    print(iou_matrix)

    cost_matrix = 1 - iou_matrix

    # best mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    print(row_ind, col_ind)
    # mIoU
    miou = iou_matrix[row_ind, col_ind].mean()
    print("mIoU:", miou)
    overall_iou.append(miou)

    CM = np.zeros((GT_LABEL_SIZE, GT_LABEL_SIZE), dtype=np.int32)

    for i in range(adjusted_ground_truth_labels.shape[0]):
        CM[adjusted_ground_truth_labels[i] - 1, col_ind[filtered_unsupervised_labels[i] - 1]] += 1

    F1_scores = []
    IoUs = []

    for i in range(GT_LABEL_SIZE):
        TP = CM[i, i]
        FP = np.sum(CM[:, i]) - TP
        FN = np.sum(CM[i, :]) - TP

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0

        F1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        F1_scores.append(F1)

        iou = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0
        IoUs.append(iou)

    accuracy = np.sum(np.diag(CM)) / np.sum(CM)
    mIoU = np.mean(IoUs)
    mF1 = np.mean(F1_scores)
    overall_accuracy.append(accuracy)
    overall_f1_score.append(mF1)
    overall_iou2.append(mIoU)

    fig = plt.figure(figsize=(7, 7))
    ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    print(np.unique(col_ind[filtered_unsupervised_labels - 1]))
    for label in np.unique(col_ind[filtered_unsupervised_labels - 1]):
        ax2.scatter(filtered_point_cloud[col_ind[filtered_unsupervised_labels - 1] == label, 0],
                    filtered_point_cloud[col_ind[filtered_unsupervised_labels - 1] == label, 1],
                    filtered_point_cloud[col_ind[filtered_unsupervised_labels - 1] == label, 2], linestyle='None')
    ax2.axis('off')
    ax2.set_facecolor("none")
    fig.tight_layout()
    fig.patch.set_facecolor("none")
    ax2.grid(False)
    if os.path.exists(f'{method}') == False:
        os.mkdir(f'{method}')
    fig.savefig(f'{method}/{filename}_matching.pdf')
    fig.clf()

    fig = plt.figure(figsize=(7, 7))
    ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    for label in np.unique(adjusted_ground_truth_labels - 1):
        ax2.scatter(filtered_point_cloud[adjusted_ground_truth_labels - 1 == label, 0],
                    filtered_point_cloud[adjusted_ground_truth_labels - 1 == label, 1],
                    filtered_point_cloud[adjusted_ground_truth_labels - 1 == label, 2], linestyle='None')
    ax2.axis('off')
    ax2.set_facecolor("none")
    fig.tight_layout()
    fig.patch.set_facecolor("none")
    ax2.grid(False)
    if os.path.exists(f'{method}') == False:
        os.mkdir(f'{method}')
    fig.savefig(f'{method}/{filename}_gt.pdf')
    fig.clf()
    # rematch visualization
    if VIS:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])
        point_cloud_inferred = o3d.geometry.PointCloud()
        point_cloud_inferred.points = o3d.utility.Vector3dVector(inference)
        point_cloud_inferred.colors = o3d.utility.Vector3dVector(label_color[col_ind[inference_labels - 1] + 1])
        vis.add_geometry(point_cloud_inferred)
        vis.run()
        screenshot = vis.capture_screen_float_buffer(False)
        output_image_path = 'C:\\Users\\Tung\\Desktop\\repos\\dataset-devkit\\visualization'
        o3d.io.write_image(os.path.join(output_image_path, filename + "_inference.png"), screenshot)
        vis.destroy_window()

print("Test mIoU", np.mean(overall_iou))
print("Test mIoU2", np.mean(overall_iou2))
print("Test Accuracy", np.mean(overall_accuracy))
print("Test F1-Score", np.mean(overall_f1_score))
