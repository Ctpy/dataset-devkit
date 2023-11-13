# Data

The following data used are

- `test_sampled[datset]`: Contains the sampled point cloud data
- `test_sampled_label[dataset]`: Contains the corresponding sampled label data
- `kitti_object_test-v0.1.json`: Contains the manually labelled test set data

## Manually labeled test set

The important key-value pairs in the `json` files are:

- `categories`: specifying the labels e.g. `front`, `left`, `right`, `top`, `back` and `undefined`.
- `samples/name`: specifying the test data corresponding to the kitti data
- `samples/point_annotations`: gt of test labels
