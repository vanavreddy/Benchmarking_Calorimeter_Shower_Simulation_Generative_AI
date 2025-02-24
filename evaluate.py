"""
This code is  partially adopted from CaloChallenge github page. Here is the link for it
https://github.com/CaloChallenge/homepage/blob/main/code/

"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import HighLevelFeatures as HLF
from utils import *
import re
from matplotlib import gridspec
from scipy.stats import wasserstein_distance
from evaluate_metrics_helper import *
import configargparse
import jetnet
from pearson_frob import *
from evaluate_range import *




    
def evaluate_metrics_ds_2_3(Es, Showers, HLFs, model_names, files, args,model_to_color_dict):
    """Plot histograms for dataset 2 and 3 """
    #compute layer-wise energy distribution:
    plot_layers(args,model_to_color_dict)
    ## pearson_frob_calc:
    calc_CFD(Showers, model_names, args.output_dir,args.dataset_num,model_to_color_dict)
    #sparsity
    plot_sparsity_group(HLFs, args.dataset_num,args.output_dir,args.particle_type, model_names,model_to_color_dict, width=4,height=3,
                        TITLE_SIZE=args.title_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                                LEGEND_SIZE=args.legend_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    min_energy=0.5e-3/0.033
    #voxel energy dist
    plot_cell_dist(Showers,min_energy,args.dataset_num,args.output_dir, args.particle_type, model_names,model_to_color_dict, 
                      ratio = False,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #E_ratio
    plot_Etot_Einc_new(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names,model_to_color_dict, 
                      row=1,col=1,height=6,width=8,YMAX=20,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #center of energy in eta direction
    plot_ECEtas_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, model_to_color_dict,
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

    #center of energy in phi direction
    plot_ECPhis_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, model_to_color_dict,
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #shower width in eta
    plot_SW_etas_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, model_to_color_dict,
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #shower width in phi
    plot_SW_Phis_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, model_to_color_dict, 
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    
def evaluate_metrics_ds_1(Es, Showers, HLFs, model_names, files, args):
    """ Plot histograms for dataset 1"""
    plot_Etot_Einc_new(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                              row=1,col=1,height=6,width=8,YMAX=25,LEGEND_SIZE=args.legend_size,
                              XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                              TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    min_energy=10

    plot_cell_dist(Showers,min_energy,args.dataset_num,args.output_dir,args.particle_type, model_names, width=7,height=4,
                       TITLE_SIZE=args.title_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
           LEGEND_SIZE=args.legend_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size,YMAX=3,ratio = False)
    plot_ECEtas(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                ratio = False,row=args.row,col=args.col,height=4,width=6,

                YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    plot_ECPhis(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                ratio = False,row=args.row,col=args.col,height=4,width=6,

                YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

    plot_SW_Phis(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                    row=args.row,col=args.col,height=4,width=6,

                    YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

    if args.particle_type=='photon':
        plot_SW_Etas(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                    ratio = False,row=args.row,col=args.col,height=4,width=6,

                    YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    elif args.particle_type=='pion':
        plot_SW_Etas_pion(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                    ratio = False,row=args.row,col=args.col,height=4,width=6,

                    YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

def evaluate_fpd_kpd(Es,Showers,HLFs,model_names,files,args):
    """Calculates FPD and KPD scores """
    fpd_vals={}
    kpd_vals={}
    fpd_errs={}
    kpd_errs={}
    g_index=model_names.index('Geant4')
            
    reference_HLF=HLFs[g_index]
    reference_file = h5py.File(files[g_index],'r')
    reference_array = prepare_high_data_for_classifier(reference_file, reference_HLF,1)
    reference_array=reference_array[:, :-1]

    for j in range(len(model_names)):
        if j!=g_index:
            source_file=h5py.File(files[j],'r')
            source_array= prepare_high_data_for_classifier(source_file, HLFs[j],0)
            source_array=source_array[:, :-1]
            fpd_val, fpd_err = jetnet.evaluation.fpd(reference_array, source_array)
            kpd_val, kpd_err = jetnet.evaluation.kpd(reference_array, source_array)
            name="dataset_"+str(args.dataset_num)+"_particle_"+args.particle_type+"_model_names"+model_names[j]
            fpd_vals[name]=fpd_val
            kpd_vals[name]=kpd_val
            fpd_errs[name]=fpd_err
            kpd_errs[name]=kpd_err
        print("done with:", model_names[j])

    write_dict_to_txt(fpd_vals,"fpd_val_"+str(args.dataset_num)+"_"+args.particle_type+".txt")
    write_dict_to_txt(kpd_vals,"kpd_val_"+str(args.dataset_num)+"_"+args.particle_type+".txt")

    write_dict_to_txt(fpd_errs,"fpd_errs_"+str(args.dataset_num)+"_"+args.particle_type+".txt")
    write_dict_to_txt(kpd_errs,"kpd_errs_"+str(args.dataset_num)+"_"+args.particle_type+".txt")
    

    
    
def main():
    
    ### ......input arguments....
    args = parse_arguments()
    print("printing all arguments.....\n")
    print(args)
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.metrics=='all':
         ## this returns incident energy, showers, HLFs object for each sample dataset and their order in the folder
        Es,Showers,HLFs,model_names,files=initialize_HLFs(args.dataset_path,args.particle_type,args.binning_file)
        model_to_color_dict=create_model_to_color_dict(model_names)
    
        if args.dataset_num==2 or args.dataset_num== 3:
            #computing fpd_kpd
            evaluate_fpd_kpd(Es,Showers,HLFs,model_names,files,args)
            evaluate_metrics_ds_2_3(Es, Showers, HLFs, model_names, files, args,model_to_color_dict)
            
        elif args.dataset_num==1:
            evaluate_metrics_ds_1(Es, Showers, HLFs, model_names, files, args)
        else:
            print(f"Error in {args.dataset_num}.")
            
    elif args.metrics=='sep':
        taskname='separation_power'
        plot_sep_emd(args.sep_file_path, args.output_dir, args.dataset_num, args.particle_type,width=7,height=5,taskname=taskname)
    elif args.metrics=='emd':
        taskname='emd_score'
        plot_sep_emd(args.sep_file_path, args.output_dir, args.dataset_num, args.particle_type,width=7,height=5,taskname=taskname)
        
    elif args.metrics=='fpd-kpd':
        
        Es,Showers,HLFs,model_names,files=initialize_HLFs(args.dataset_path,args.particle_type,args.binning_file)
        evaluate_fpd_kpd(Es,Showers,HLFs,model_names,files,args)
    elif args.metrics=='CFD':
        Es,Showers,HLFs,model_names,files=initialize_HLFs(args.dataset_path,args.particle_type,args.binning_file)
        model_to_color_dict=create_model_to_color_dict(model_names)
        calc_CFD(Showers, model_names, args.output_dir,args.dataset_num,model_to_color_dict)
        
    elif args.metrics=='layer':
        plot_layers(args)
        
    else:
        print(f"Error! {args.metrics} is not implemented.")
        
        


if __name__ == "__main__":
    main()
