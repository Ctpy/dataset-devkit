# Dataset-Devkit

This is the dataset devkit which enables to perform:
- Visualization of point cloud data
- Transformation of point cloud data
- Reading different datasets
- Evaluation of custom unsupervised segmentation performance
- Creating single object datasets

## Creating datasets

To create a dataset of single objects run:

```bash
python create_dataset.py ${dataset} --input_path ${input_path} --output_path ${output_path} --worker ${worker}
```

This will create a dataset of bin files with the single objects.

You can also adapt that to generate the output format to `.pcd`

