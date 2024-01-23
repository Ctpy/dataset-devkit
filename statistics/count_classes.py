import json

data = json.load(open('C:\\Users\\Tung\\repos\\dataset-devkit\\data\\kitti_object_test-v0.1.json'))

samples = data['dataset']['samples']
num_classes = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
}
num_samples = 0
num_points_per_sample = 0
num_points_per_class = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
}
for sample in samples:
    try:
        is_labeled = sample['labels']['ground-truth']['label_status'] == 'LABELED'
        if not is_labeled:
            continue
    except TypeError:
        continue

    classes = sample['labels']['ground-truth']['attributes']['annotations']
    map_id_to_class = {'0': 0}
    for class_ in classes:
        num_classes[str(class_['category_id'])] += + 1
        map_id_to_class[str(class_['id'])] = str(class_['category_id'])
    points = sample['labels']['ground-truth']['attributes']['point_annotations']
    for point in points:
        num_points_per_class[str(map_id_to_class[str(point)])] += 1
    num_points_per_sample += len(points)
    num_samples += 1
print(num_samples)
print(num_classes)
print(num_points_per_sample/64)
print(num_points_per_class)
