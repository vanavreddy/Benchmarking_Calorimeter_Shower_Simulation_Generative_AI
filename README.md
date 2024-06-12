# A Comprehensive Evaluation of Generative Models in Calorimeter Shower Simulation

* samples in a folder is saved as dataset_n_particle_model.h5 where n stands for the dataset number, partcile stands for type of particle e.g. electron, photon and pion and model stands for CaloDiffusion, CaloScore, CaloINN and Geant4. In our code we assume that samples are saved in this manner in a folder and based on that we read from a path. Upon request we can share our generated samples with the reviewers. We could not upload them now due to the file size constraints.

1. To generate Sparsity, Center of Energy, Shower width, voxel distribution, and E_ratio plots, run the following commands:

```
# Dataset 1(photon)
python evaluate.py --binning_file ‘path_to_binning_file’ --dataset_path ‘path_to_dataset_path’ --dataset_num 1 --particle_type ‘photon’ --metrics ‘all’ --row 1 --col 2

# Dataset 1 (pion):
python evaluate.py --binning_file ‘path_to_binning_file’ --dataset_path ‘path_to_dataset_path’ --dataset_num 1 --particle_type ‘pion’ --metrics ‘all’ --row 2 --col 2

# Dataset 2 and 3:
python evaluate.py --binning_file ‘path_to_binning_file’ --dataset_path ‘path_to_dataset_path’ --dataset_num 2 --particle_type ‘electron’ --metrics ‘all’ --row 3 --col 3
```

2. To generate the plots for separation_power or EMD, run the following commands:

```
python evaluate.py --metrics 'sep' --dataset_num n --particle_type 'type_of_particles' --sep_file_path 'path_to_separation.txt generated from command at 1'

python evaluate.py --metrics 'emd' --dataset_num n --particle_type 'type_of_particles' --sep_file_path 'path_to_separation.txt generated from command at 1'
```

3. To generate FPD and KPD scores, run the following commands:
```
python evaluate.py --binning_file ‘path_to_binning_file’ --dataset_path ‘path_to_dataset_path’ --dataset_num 2 --particle_type ‘electron’ --metrics ‘fpd-kpd’ 
```

4. To generate correlation plots similar to the ones publised in the paper, run the following command:
```
python correlate.py -i 'path_to_model_sampple_file' -r 'path_to_reference_sample_file' -m 'corr' -n 'model_name' -d 'dataset_num'
```

5. To generate AUC and JSD scores, run the following command:
```
python classifier_auc_jsd.py --input_file 'path_to_input_file' --reference_file 'path_to_reference_file' --dataset 'dataset_num' --mode '[cls-low, clow-low-normed, cls-high]' --binning_file 'path_to_binning_file'
```

