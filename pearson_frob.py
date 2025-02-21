import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import seaborn as sns
import argparse
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

#parsing arguments--->
parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))


parser.add_argument('--dataset_path', '-dp', default='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2',
                    help='Name of the directory to be evaluated.')

parser.add_argument('--mode', '-m', default='voxel',choices=['voxel','layer','group'],
                    help=('how the correlations will be computed.' ))

parser.add_argument('--dataset', '-d', choices=['1-photons', '1-pions', '2', '3'],
                    help='Which dataset is evaluated.')
parser.add_argument('--output_dir','-o', default='evaluation_results/',
                    help='Where to store evaluation output files (plots and scores).')
parser.add_argument('--plot','-p', default='bar_plot',
                    help='What type of plotting. options are heatmap and bar_plot')

def file_read(file_name):
    """
    argument: file name of the generated and reference data
    returns incident energy and showers.
    """
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        
    return e, shower

def grouping_data(data):
    """
    First summing along the angular bins making it an array of shape(-1,45,radial_bin)
    grouping consecutive 5 layers  
    """
    data=np.sum(data, axis=2)
    data = data.reshape(-1, 9, 5, 9)
    data = data.mean(axis=2)
    #print("in grouping data : ",data.shape)
    return data

def draw_voxel_heatmap(corr_mat_gen, corr_mat_ref, name, output_dir,model_name):
    """
    Generates a PDF with side-by-side correlation heatmaps for two datasets.
    
    Parameters:
        corr_mat_gen (list of ndarray): Correlation matrices for the generated data.
        corr_mat_ref (list of ndarray): Correlation matrices for the reference data.
        output_pdf (str): Output file name for the PDF.
        
    """
   
    if len(corr_mat_gen) != len(corr_mat_ref):
        raise ValueError("The number of layers in generated and reference data must match.")
    output_pdf = os.path.join(output_dir, name)

    # Create a PDF object to store plots
    with PdfPages(output_pdf) as pdf:
        fig_title = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5,model_name, ha='center', va='center', fontsize=20, fontweight='bold')
        plt.axis('off')  # Hide axis
        pdf.savefig(fig_title)  # Save the title page
        plt.close(fig_title)
        for idx, (gen_mat, ref_mat) in enumerate(zip(corr_mat_gen, corr_mat_ref)):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Plot for generated data
            sns.heatmap(gen_mat, ax=axes[0], cmap="coolwarm", annot=False, cbar=True,vmin=0,vmax=1)
            axes[0].set_title(f"{model_name} Data: Layer {idx+1} and {idx+2}")
            axes[0].set_xlabel("Voxel in radial bins")
            axes[0].set_ylabel("Voxel in angular bins")

            # Plot for reference data
            sns.heatmap(ref_mat, ax=axes[1], cmap="coolwarm", annot=False, cbar=True,vmin=0,vmax=1)
            axes[1].set_title(f"Geant4 Data: Layer {idx+1} and {idx+2}")
            axes[1].set_xlabel("Voxel in radial bins")
            axes[1].set_ylabel("Voxel in angular bins")

            # Adjust layout and save to PDF
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
def draw_heatmap(correlation_matrix,name,output_dir,width=8,height=6,TITLE_SIZE=30
                   ,XLABEL_SIZE=25,YLABEL_SIZE=25,LEGEND_SIZE=16,XTICK_SIZE=24,YTICK_SIZE=24,STEPSIZE=5,CBAR=True):
    
    """
    This is a simple function to fraw the heatmap for layer wise correlation
    """
    sns.set()  # Set seaborn style
    fig,ax = plt.subplots(figsize=(width, height))  # Set the figure size
    
    
    sns.heatmap(correlation_matrix, ax=ax,annot=False, cmap='coolwarm', fmt='.2f',cbar=CBAR,vmin=0,vmax=1)
    ax.set_xlabel('Layers',fontsize=XLABEL_SIZE)
    ax.set_ylabel('Layers',fontsize=YLABEL_SIZE)
    ax.tick_params(axis='x', rotation=90,labelsize=XTICK_SIZE)
    ax.tick_params(axis='y', rotation=0,labelsize=YTICK_SIZE)
   
    if CBAR:
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=LEGEND_SIZE)
    plt.gca().invert_yaxis()
    
    fig.savefig(name, bbox_inches='tight',dpi=350)  # Save the figure
    # Show the heatmap
    save_path = os.path.join(output_dir, name)
    plt.savefig(save_path)


def generate_group_correlation_matrices(data):
    """
    Generate voxel-wise correlation matrices for consecutive layers.
    
    Parameters:
        data (ndarray): Input data of shape (100000, group#, radial_bins).
        
    Returns:
        correlation_matrices (list of ndarray): A list of correlation matrices  for each layer pair.
    """
    n_layers = data.shape[1]
    correlation_matrices = []
   
    r_bin=data.shape[2]
    # Iterate over each pair of consecutive layers
    for i in range(n_layers - 1):
        # Extract data for the two layers
        layer_data_1 = data[:, i, :].reshape(-1, r_bin)  # Shape: (100000, 16, 9)
        layer_data_2 = data[:, i + 1, :].reshape(-1,  r_bin)  # Shape: (100000, 16, 9)

        # Initialize the correlation matrix for this layer pair
        correlation_matrix = np.zeros((9))

        # Compute voxel-wise correlations

        for col in range(r_bin):
            corr, _ = pearsonr(layer_data_1[:, col], layer_data_2[:, col])
            correlation_matrix[col] = corr

        correlation_matrices.append(correlation_matrix)

    return np.array(correlation_matrices)


def extract_model_names(evaluate_files_list):
    """
    A helper function to read the file name. Remember samples are saved in a pattern of 
    'dataset_X_PARTICLE_MODEL.h5' or 'dataset_X_PARTICLE_MODEL.hdf5'
    This function will find out the available types of model. Make sure you save the Reference data as 
    'dataset_2_electron_Geant4.h5'
    
    """
    model_names = []
    for name in evaluate_files_list:
        m_name = name.split('/')[-1].split('.')[0].split('_')[-1]
        if m_name=='Geant4':
            idx=len(model_names)
        model_names.append(m_name)

    return model_names, idx

def calculate_frob_norm(corr_geant, corr_gen):
    """
    computing frobenius norm between two numpy array manually.
    """
    return np.sqrt(np.sum((corr_geant - corr_gen) ** 2))


def calculateCorrelation(layer_data):
    """
    computing correlation between layer i and layer i+1.
    """
    dim=layer_data.shape[1]
    correlation_matrix = np.ones((dim, dim))
    p_value_matrix = np.zeros((dim, dim))

    # Loop through each pair of layers and compute correlations and p-values
    for i in range(dim):
        for j in range(dim):
            if i != j:  # Exclude self-correlation (diagonal elements)
                corr, p_value = pearsonr(layer_data[:, i], layer_data[:, j])
                correlation_matrix[i, j] = corr
                p_value_matrix[i, j] = p_value
                
                
    return correlation_matrix,p_value_matrix

def generate_correlation_matrices(data):
    """
    Generate voxel-wise correlation matrices for consecutive layers.
    That means, correlation between layer i and layer i+1 for voxel j
    
    Parameters:
        data (ndarray): Input data of shape (100000, 45, 16, 9).
        
    Returns:
        correlation_matrices (list of ndarray): A list of correlation matrices (16x9) for each layer pair. Final shape is (44,16,9)
    """
    n_layers = data.shape[1]
    correlation_matrices = []
    a_bin=data.shape[2]
    r_bin=data.shape[3]
    # Iterate over each pair of consecutive layers
    for i in range(n_layers - 1):
        # Extract data for the two layers
        layer_data_1 = data[:, i, :, :].reshape(-1, a_bin,r_bin)  # Shape: (100000, 16, 9)
        layer_data_2 = data[:, i + 1, :, :].reshape(-1, a_bin,r_bin)  # Shape: (100000, 16, 9)

        # Initialize the correlation matrix for this layer pair
        correlation_matrix = np.zeros((16, 9))

        # Compute voxel-wise correlations
        for row in range(a_bin):
            for col in range(r_bin):
                corr, _ = pearsonr(layer_data_1[:, row, col], layer_data_2[:, row, col])
                correlation_matrix[row, col] = corr

        correlation_matrices.append(correlation_matrix)

    return np.array(correlation_matrices)


def plot_frob_norm(frobs,name,model_names,output_dir,mode):
    

    # Plotting frobenius norm as a bar_plot
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, frobs, color='skyblue', edgecolor='black')

    # Customize the plot
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Frobenius Norm with Geant4", fontsize=12)
    plt.title(f"Frobenius Norm Comparison with Geant4 for {mode} wise correlation", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'frob_norm_{name}')
    plt.savefig(save_path)
   
    
def draw_plots(correlations,g_idx,name,model_names,out_dir,mode):
    frobs=[]
    re_models=[]
    corr_g=correlations[g_idx]
    

    #if plot=='heatmap':
    
    for i,corr in enumerate(correlations):
        file_name='heatmap_'+model_names[i]+'_'+name
        if mode=='voxel':
            draw_voxel_heatmap(corr,correlations[g_idx],file_name,out_dir,model_names[i])
        elif mode=='layer':
            draw_heatmap(corr,file_name,output_dir=out_dir)
        else: 
            print("not implemented yet")
                
            
    #elif plot=='bar_plot':
    for i,corr in enumerate(correlations):
        if i!=g_idx:
            f=calculate_frob_norm(corr_g, corr)
            frobs.append(f)
            re_models.append(model_names[i])
    plot_frob_norm(frobs,name, re_models,out_dir,mode)

    # else:
    #     print("will be updated later!")
def calc_CFD(Showers, model_names, out_dir,dataset):
    
    g_idx=model_names.index('Geant4')
    if dataset==2:
        shape=[-1,45,16,9]
    elif dataset==3:
        shape=[-1,45,50,18]
    else:
        print('Not implemented yet for dataset 1 photon and pion')
        return
    
    mode='voxel'
    ## looking at the correlation between the voxel j of layer i and the voxel j of layer i+1
    correlations=[]

    for S in Showers:
        correlations.append(generate_correlation_matrices(S.reshape(shape)))
    fileName=str(dataset)+'_'+mode+'.pdf'
    draw_plots(correlations,g_idx,fileName,model_names,out_dir,mode)

    mode='layer'
    ## looking at the correlation between the layer i and layer i+1, considering their layer_sum
    correlations=[]
    for S in Showers:
        summ=S.reshape(shape).sum(axis=(2,3))
        corr,_=calculateCorrelation(summ)
        correlations.append(corr)
    fileName=str(dataset)+'_'+mode+'.pdf'    
    draw_plots(correlations,g_idx,fileName,model_names,out_dir,mode)

    mode='group'
    ## First create a group of layers by combining 5 consecutive layers, then sum along the axis of angular bins  and
    ## finally compute correlation between group i's radial_bin j with group i+1's radial_bin j
    correlations=[]

    for S in Showers:
        data=grouping_data(S.reshape(shape))
        correlations.append(generate_group_correlation_matrices(data))
    fileName=str(dataset)+'_'+mode+'.pdf'   
    draw_plots(correlations,g_idx,fileName,model_names,out_dir,mode)

  


def gen_CFD(evaluate_path, dataset,out_dir):
    
    #locating all the generated and reference files in the dataset_path
    evaluate_files_list = [
    str(p) for ext in ('*.h5', '*.hdf5') for p in Path(evaluate_path).rglob(ext)
    ]
    
    #get the model_names list and the index of Geant4 data in the list
    model_names,g_idx = extract_model_names(evaluate_files_list)
    
    Energies=[]
    Showers=[]
    if dataset==2:
        shape=[-1,45,16,9]
    elif dataset==3:
        shape=[-1,45,50,18]
    else:
        print('Not implemented yet for dataset 1 photon and pion')
    #reading the incident energies and showers for further analysis.        
    for eval_file in evaluate_files_list:
        
        E, S=file_read(eval_file)
        Energies.append(E)
        S=S.reshape(shape)
        Showers.append(S)
        
        
    mode='voxel'
    ## looking at the correlation between the voxel j of layer i and the voxel j of layer i+1
    correlations=[]

    for S in Showers:
        correlations.append(generate_correlation_matrices(S))
    fileName=str(dataset)+'_'+mode+'.pdf'
    draw_plots(correlations,g_idx,fileName,model_names,out_dir,mode)
            
    mode='layer'
    ## looking at the correlation between the layer i and layer i+1, considering their layer_sum
    correlations=[]
    for S in Showers:
        summ=S.sum(axis=(2,3))
        corr,_=calculateCorrelation(summ)
        correlations.append(corr)
    fileName=str(dataset)+'_'+mode+'.pdf'    
    draw_plots(correlations,g_idx,fileName,model_names,out_dir,mode)
        
    mode='group'
    ## First create a group of layers by combining 5 consecutive layers, then sum along the axis of angular bins and
    ## finally compute correlation between group i's radial_bin j with group i+1's radial_bin j
    correlations=[]

    for S in Showers:
        data=grouping_data(S)
        correlations.append(generate_group_correlation_matrices(data))
    fileName=str(dataset)+'_'+mode+'.pdf'   
    draw_plots(correlations,g_idx,fileName,model_names,out_dir,mode)
        
        
    
            
        
if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
                          
    gen_CFD(args.dataset_path,args.dataset,args.output_dir)
    
        
        
        
        
        