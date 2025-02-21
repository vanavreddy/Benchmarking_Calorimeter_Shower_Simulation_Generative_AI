# Benchmarking_Calorimeter_Shower_Simulation_Generative_AI

In this project we are comparing 4 different Generative AI models on Calorimeter Shower Simulation. This project is inspired by CaloChallenge 2022[https://calochallenge.github.io/homepage/].

## Dataset
Dataset 1 available at https://zenodo.org/records/8099322
Dataset 2 available at https://zenodo.org/records/6366271
Dataset 3 available at https://zenodo.org/records/6366324
You can access our generated samples here: [Dataset on Zenodo](https://zenodo.org/records/14883798).


## Generative AI models

Here we compare 4 different Generative AI models.
They are 
1. CaloDream on dataset 2 and 3 (Conditional Flow Matching)https://github.com/luigifvr/calo_dreamer
2. CaloDiffusion (Denoising Diffusion based model) https://github.com/OzAmram/CaloDiffusionPaper/tree/main
3. CaloScore (Score based model) https://github.com/ViniciusMikuni/CaloScoreV2/tree/main
4. CaloINN (Combination of VAE and Normalizing Flow) https://github.com/heidelberg-hepml/CaloINN/tree/calochallenge

## Directory structure

- The main evaluation scripts (and helper modules) to generate plots and various metrics are at the top level.
- The 'xml_binning_files' folder contains the binning file in XML needed to run the evaluation scripts.
- The 'trained_models' folder contains the pre-trained models, including CaloDiffusion, CaloScore, CaloINN and Geant4, for each dataset.

A suitable python environment named eval can be created and activated with:
```
python -m venv eval
source eval/bin/activate
pip install -r requirements.txt

```



| Parameters    | Usage |
|:------------:|:------|
| dataset_path | Path to the folder that contains samples from different models and Geant4. Files are saved in a pattern of 'dataset_NUM_PARTICLE_MODEL.h5'. |
|sep_file_path| Path to the separation or emd score files.|
| metrics         | Options: 'all', 'fpd-kpd', 'sep', 'emd'  |
| dataset_num      | Type of dataset:1,2,3 |
| output_dir   | Path to the directory to save results. |
| binning_file  | Path to binning file. |
|particle_type  |  Type of the particle being evaluated e.g., photon, pion, electron.|


## Running the evaluation scripts

1. To generate Sparsity, Center of Energy, Shower width, voxel distribution, and E_ratio plots, run the following commands:

```
# Dataset 1(photon)
python evaluate.py --binning_file 'path_to_binning_file' --dataset_path 'path_to_dataset_path' --dataset_num 1 --particle_type 'photon' --metrics 'all' --row 1 --col 2

# Dataset 1 (pion):
python evaluate.py --binning_file 'path_to_binning_file' --dataset_path 'path_to_dataset_path' --dataset_num 1 --particle_type 'pion' --metrics 'all' --row 2 --col 2

# Dataset 2 and 3:
python evaluate.py --binning_file 'path_to_binning_file' --dataset_path 'path_to_dataset_path' --dataset_num '[2, 3]' --particle_type 'electron' --metrics 'all' --row 3 --col 3
```

2. To generate the plots for separation_power or EMD, run the following commands:

```
python evaluate.py --metrics 'sep' --dataset_num n --particle_type 'type_of_particles' --sep_file_path 'path_to_separation.txt generated after running previous command (1)'

python evaluate.py --metrics 'emd' --dataset_num n --particle_type 'type_of_particles' --sep_file_path 'path_to_separation.txt generated after running previous command (1)'
```

3. To generate FPD and KPD scores, run the following commands:
```
python evaluate.py --binning_file ‘path_to_binning_file’ --dataset_path ‘path_to_dataset_path’ --dataset_num 'dataset_num' --particle_type ‘electron’ --metrics ‘fpd-kpd’ 
```

4. To generate correlation plots similar to the ones publised in the paper, run the following command:
```
python evaluate.py --dataset_path ‘path_to_dataset_path’ --dataset_num  2/3 --output_dir 'path_to_output_directory'
```

5. To generate AUC and JSD scores, run the following command:
```
python classifier_auc_jsd.py --input_file 'path_to_input_file' --reference_file 'path_to_reference_file' --dataset 'dataset_num' --mode '[cls-low, clow-low-normed, cls-high]' --binning_file 'path_to_binning_file'
```

Note: The samples in a given folder are saved with specific naming convension. Specifically, dataset_n_particle_model.h5, where n stands for the dataset number, partcile stands for type of particle, e.g., electron, and model stands for CaloDiffusion, CaloScore, CaloINN or Geant4. In our evaluation scripts, we assume the saved samples follow this naming convension and based on that we read from the path. Upon request we can share our generated samples with the reviewers. We could not upload them now due to the file size constraints.

