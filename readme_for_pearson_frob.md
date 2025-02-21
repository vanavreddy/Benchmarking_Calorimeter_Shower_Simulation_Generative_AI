## Required packages:

- numpy
- h5py
- matplotlib
- scipy
- seaborn
- jetnet
- configargparse

## Dataset
You can access the dataset here: [Dataset on Zenodo](https://zenodo.org/records/14883798).

| Parameters    | Usage |
|:------------:|:------|
| dataset_path | Path to the folder that contains samples from different models and Geant4. Files are saved in a pattern of 'dataset_NUM_PARTICLE_MODEL.h5'. |
| mode         | How do we want to compute the correlation. Options are voxel, layer, and group. |
| dataset      | Type of dataset:'2', '3'. |
| output_dir   | Path to the directory to save results. |
| plot         | What kind of plot we want to generate: bar_plot or heatmap. |

To plot the Frobenius norm of the computed Pearson correlation coefficient as a bar plot, run the following command:

```
python pearson_frob.py -m voxel -d 2 -p bar_plot -dp PATH_TO_SAMPLES -o PATH_TO_STORE_RESULTS
```

To plot the intermediate correlation matrices for mode voxel and layer as a heatmap, run the following command:

```
python pearson_frob.py -m voxel -d 2 -p heatmap -dp PATH_TO_SAMPLES -o PATH_TO_STORE_RESULTS
```
