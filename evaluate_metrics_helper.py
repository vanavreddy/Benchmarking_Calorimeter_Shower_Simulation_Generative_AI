
"""
This code is partially adopted from CaloChallenge github page. Here is the link for it
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
import math



def plot_cell_dist(shower_arr,min_energy,dataset,output_dir, particle,model_names, width=8,height=5,TITLE_SIZE=30
                   ,XLABEL_SIZE=25,YLABEL_SIZE=25,YMAX=1,ratio = False,LEGEND_SIZE=24,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots voxel energies across all layers """
    x_scale='log'
    EMDs={}
    Seps={}
    legend_names=['Geant4']
    g_index=model_names.index('Geant4')
    ref_shower_arr=shower_arr[g_index] # trying to convert to GeV
    fig0, ax0 = plt.subplots(1,1,figsize=(width*1,height*1),sharex=True,sharey=True)
    
    

    if x_scale == 'log':
        bins = np.logspace(np.log10(min_energy),
                           np.log10(ref_shower_arr.max()),
                           50)
    else:
        bins = 50

    eps = 1e-16
    counts_ref, _, _ = ax0.hist(ref_shower_arr.flatten(), bins=bins,
                                color = model_to_color_dict[model_names[g_index]],
                                label='Geant4', density=True, histtype='step',
                                alpha=1.0, linewidth=3.)
    
    for j in range(len(shower_arr)):
        if j==g_index:
            continue
        legend_names.append(model_names[j])
        counts_data, _, _ = ax0.hist(shower_arr[j].flatten() + eps, label=model_names[j], bins=bins, 
                                     color = model_to_color_dict[model_names[j]],
                                     histtype='step', linewidth=3., alpha=0.5, density=True)
        
        
        emd_score=getEMD(counts_ref,counts_data)
        EMDs[model_names[j]]=emd_score
        
        seps = _separation_power(counts_ref, counts_data, bins)
        Seps[model_names[j]]=seps

        ax0.set_ylabel('A.U.',fontsize=YLABEL_SIZE)
        ax0.set_ylim([None,YMAX])
        #ax0.set_yscale('log')
        ax0.margins(0.05, 0.5)
        ax0.tick_params(axis='x', labelsize=XTICK_SIZE)
        ax0.tick_params(axis='y', labelsize=YTICK_SIZE)
        #ax0.set_ylabel("Arbitrary units")
        #plt.xlabel(r"Voxel Energy [MeV]")
        ax0.set_yscale('log')
        if x_scale == 'log': ax0.set_xscale('log')
    #plt_label = "Voxel Energy Distribution for Dataset "+dataset
    #fig0.suptitle(plt_label,y=1.1,fontsize=TITLE_SIZE) # for positioning title
    fig0.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig0.legend(lines, labels, loc='upper center',ncol = 4,fontsize=LEGEND_SIZE,bbox_to_anchor=[0.55, 1.02]) #for positioning figure
    fig0.legend(legend_names[:len(model_names)],fontsize=LEGEND_SIZE,loc='upper center', bbox_to_anchor=[0.5, 1.06],ncol=4,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2)
    filename0 = os.path.join(output_dir, 'E_voxel_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350,bbox_inches='tight')
    emd_file=os.path.join(output_dir,"emd_E_voxel_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir,"separation_E_voxel_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()

    
    
def plot_Etot_Einc_new(HLFs,dataset,output_dir, particle, model_names, ratio = False,row=1,col=1,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots Etot normalized to Einc histogram """
    EMDs={}
    Seps={}
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=True)
    # if("pion" in dataset):
    #     xmin, xmax = (0., 2.0)
    # else:
    legend_names=['Geant4']
    xmin, xmax = (0.5, 1.5)
    g_index=model_names.index('Geant4')
    reference_class=HLFs[g_index]
    bins = np.linspace(xmin, xmax, 101)
  

    counts_ref, _, _ = ax0.hist(reference_class.GetEtot() / reference_class.Einc.squeeze(),
                                bins=bins, label='Geant4', density=True, color = model_to_color_dict[model_names[g_index]],
                                histtype='step', alpha=1.0, linewidth=3.)
    
    for j in range(len(HLFs)):
        if j == g_index:
            continue
        legend_names.append(model_names[j])
        counts_data, _, _ = ax0.hist(HLFs[j].GetEtot() / HLFs[j].Einc.squeeze(), bins=bins,color =model_to_color_dict[model_names[j]] ,
                                     label=model_names[j], histtype='step', linewidth=3., alpha=0.8,
                                     density=True)
        emd_score=getEMD(counts_ref,counts_data)
        EMDs[model_names[j]]=emd_score
        
        seps=_separation_power(counts_ref,counts_data,bins)
        Seps[model_names[j]]=seps

        ax0.set_ylabel('A.U.',fontsize=YLABEL_SIZE)
        #ax0.set_xlabel(r"$E_{total}/E_{inc}$",fontsize=XLABEL_SIZE)

        ax0.set_ylim([None,YMAX])
        #ax0.set_yscale('log')
        ax0.margins(0.05, 0.5) 
        ax0.tick_params(axis='x', labelsize=XTICK_SIZE)
        ax0.tick_params(axis='y', labelsize=YTICK_SIZE)
    plt_label = "$E_{ratio}$ for Dataset "+str(dataset)
    #fig0.suptitle(plt_label,y=1.1,fontsize=TITLE_SIZE) # for positioning title
    fig0.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig0.legend(legend_names[:3], loc='upper center',ncol = 4,fontsize=LEGEND_SIZE,bbox_to_anchor=[0.5, 1.02]) #for positioning figure
    fig0.legend(legend_names[:len(model_names)],fontsize=LEGEND_SIZE,loc='upper center', bbox_to_anchor=[0.5, 1.06],ncol=4,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2)
    filename0 = os.path.join(output_dir, 'E_ratio_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350,bbox_inches='tight')


    emd_file=os.path.join(output_dir,"emd_E_ratio_dataset_{}_particle_{}.txt".format( str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir,"separation_E_ratio_dataset_{}_particle_{}.txt".format( str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()

def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()


def getEMD(dist1, dist2):
    # compute the emd score aka wasserstein distance-1
    emd_score=wasserstein_distance(dist1,dist2)
    return emd_score


def write_dict_to_txt(dictionary, filename):
    # helper function to write a dictionary to a .txt file
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")
            
            
def plot_sparsity_group(list_hlfs, dataset,output_dir, particle, model_names, ratio = False,row=3,col=3,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=False)
    """
        generates plots of sparsity distribution for dataset 2 and 3.
    """
    EMDs={}
    Seps={}
    gkeys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    dataset=str(dataset)
    legend_names=['Geant4']
    for out_idx,keys in enumerate(gkeys):
       
        if dataset in ['2', '3']:
            lim = (0, 1.0)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        
        g_index=model_names.index('Geant4')
        
        reference_class=list_hlfs[g_index]
        
        
        shape_a=reference_class.GetSparsity()[0].shape[0]

        selected_ref = [(1-reference_class.GetSparsity()[i]).reshape(shape_a,1) for i in keys]#turning into GeV
        combined_ref = np.concatenate(selected_ref, axis=1)

        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
        mean_ref = mean_ref.flatten()
        
        main_label = model_names[g_index] if out_idx==0 else None
        
        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, bins=bins,color = model_to_color_dict[model_names[g_index]],
                                    label=main_label, density=True, histtype='step',
                                    alpha=1.0, linewidth=3.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                legend_names.append(model_names[i])
                
                shape_a=reference_class.GetSparsity()[0].shape[0]

                selected_ref = [(1-list_hlfs[i].GetSparsity()[j]).reshape(shape_a,1) for j in keys]#turning into GeV
                combined_ref = np.concatenate(selected_ref, axis=1)

                mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
                mean_ref = mean_ref.flatten()
                
                sub_label = model_names[i] if out_idx==0 else None
        
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, label=sub_label, bins=bins, color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=2., alpha=0.8, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=seps
                
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {} - {}".format(keys[0],keys[4]),fontsize=XLABEL_SIZE)
            #plt.xlabel(r'[mm]')
            #ax0[out_idx//col][out_idx%col].set_xlim(*lim)
            #ax0[out_idx//col][out_idx%col].set_ylim([0,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5) 
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)
    
        
    plt_label = "Sparsity for Dataset "+str(dataset)
    #fig0.suptitle(plt_label,y=1.1,fontsize=TITLE_SIZE) # for positioning title
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'Sparsity_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
   
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig0.legend(legend_names[:3], loc='upper center',bbox_to_anchor=[0.55, 1.06],ncol=4,fontsize=LEGEND_SIZE,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2) #for positioning figure
    
    fig0.savefig(filename0, dpi=350,bbox_inches='tight')
    
    emd_file=os.path.join(output_dir,"emd_sparsity_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir,"separation_sparsity_dataset_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()

def plot_ECEtas_group(list_hlfs, dataset,output_dir, particle, model_names, ratio = False,row=3,col=3,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots center of energy in eta for dataset 2 and 3"""
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=True)
  
    if ratio:
        fig1, ax1 = plt.subplots(row,col,figsize=(width*col,height*row))
    
    EMDs={}
    Seps={}
    legend_names=['Geant4']
    gkeys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    for out_idx,keys in enumerate(gkeys):
       
        if dataset in ['2', '3']:
            lim = (-45., 45.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        
        g_index=model_names.index('Geant4')
        
        reference_class=list_hlfs[g_index]

        shape_a=reference_class.GetECEtas()[0].shape[0]

        selected_ref = [reference_class.GetECEtas()[i].reshape(shape_a,1) for i in keys]#turning into GeV
        combined_ref = np.concatenate(selected_ref, axis=1)
        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
  
        
        main_label = model_names[g_index] if out_idx==0 else None
        
        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, bins=bins,color = model_to_color_dict[model_names[g_index]],
                                    label=main_label, density=True, histtype='step',
                                    alpha=1.0, linewidth=3.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:

                legend_names.append(model_names[i])
                shape_a=list_hlfs[i].GetECEtas()[0].shape[0]

                selected_ref = [list_hlfs[i].GetECEtas()[j].reshape(shape_a,1) for j in keys]#turning into GeV
                combined_ref = np.concatenate(selected_ref, axis=1)

                mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
                
                sub_label = model_names[i] if out_idx==0 else None
                
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, label=sub_label, bins=bins, 
                                                                        color =model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=3., alpha=0.8, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=seps
                # this if blocks generates a difference plot for the histograms
                if(ratio):
                    eps = 1e-8
                    h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

                    ax1[out_idx//col][out_idx%col].axhline(y=0.0, color='black', linestyle='-',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=10, color='gray', linestyle='--',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=-10, color='gray', linestyle='--',linewidth=2)

                    xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
                    ax1[out_idx//col][out_idx%col].plot(xaxis,h_ratio,color=model_to_color_dict[model_names[i]],linestyle='-',linewidth = 3)
                    ax1[out_idx//col][out_idx%col].set_ylabel('Diff. (%)',fontsize=YLABEL_SIZE)
                    ax1[out_idx//col][out_idx%col].set_ylim([-50,50])

                    
                    
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {} - {}".format(keys[0],keys[4]),fontsize=XLABEL_SIZE)
            #plt.xlabel(r'[mm]')
            #ax0[out_idx//col][out_idx%col].set_xlim(*lim)
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5) 
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)
    
            
    plt_label = "Center of Enegery in $\\eta$ (mm) direction for Dataset "+str(dataset)
   
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
   
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE,loc='upper center', bbox_to_anchor=[0.5, 1.06],ncol=4,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2)
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'EC_eta_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    
    
    if ratio:
        fig1.tight_layout()
        filename1 = os.path.join(output_dir, 'EC_eta_dataset_{}_particle_{}_diff.pdf'.format(str(dataset),particle))
        fig1.savefig(filename1, dpi=300)
    
    emd_file=os.path.join(output_dir,"emd_EC_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir,"separation_EC_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    
    
    
    
def plot_ECEtas(list_hlfs, dataset,output_dir, particle, model_names, ratio = False,row=2,col=2,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots center of energy in eta for dataset 1(photon and pion)"""
    EMDs={}
    Seps={}
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=False,sharey=True,squeeze=False)
 
    g_index=model_names.index('Geant4')
    dataset=str(dataset)
   
    reference_class=list_hlfs[g_index]
    for out_idx, key in enumerate(reference_class.GetECEtas().keys()):
        
        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-300., 300.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)

        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(reference_class.GetECEtas()[key], bins=bins,
                                                               color = model_to_color_dict[model_names[g_index]],
                                                        label=model_names[g_index], density=True, histtype='step',
                                                        alpha=1.0, linewidth=2.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(list_hlfs[i].GetECEtas()[key], label=model_names[i], bins=bins,
                                             color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=3., alpha=0.5, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(key)]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(key)]=seps
                
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {}".format(key),fontsize=XLABEL_SIZE)
            
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5)
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)

            
    plt_label = "Center of Enegery in $\\eta$ (mm) direction for Dataset "+str(dataset)
   
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
  
    fig0.legend(lines[:len(model_names)], labels[:len(model_names)], loc='upper center',bbox_to_anchor=[0.55, 1.08],
                ncol = 4,fontsize=LEGEND_SIZE,borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
                handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2) #for positioning figure
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'EC_eta_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    new_path_emd=output_dir
    emd_file=os.path.join(new_path_emd,"emd_EC_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    new_path_sep=output_dir
    sep_file=os.path.join(new_path_sep,"separation_EC_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    
    
    
def plot_ECPhis_group(list_hlfs, dataset,output_dir, particle, model_names, ratio = False,row=3,col=3,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots center of energy in phi  for dataset 2 and 3"""
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=True)
    EMDs={}
    Seps={}
    legend_names=['Geant4']
    gkeys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    if ratio:
        fig1, ax1 = plt.subplots(row,col,figsize=(width*col,height*row))
    for out_idx,keys in enumerate(gkeys):
    
        if dataset in ['2', '3']:
            lim = (-45., 45.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        
        g_index=model_names.index('Geant4')
   
        reference_class=list_hlfs[g_index]

        shape_a=reference_class.GetECPhis()[0].shape[0]

        selected_ref = [reference_class.GetECPhis()[i].reshape(shape_a,1) for i in keys]#turning into GeV
        combined_ref = np.concatenate(selected_ref, axis=1)
        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
        main_label = model_names[g_index] if out_idx==0 else None
        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, bins=bins,color = model_to_color_dict[model_names[g_index]],
                                    label=main_label, density=True, histtype='step',
                                    alpha=1.0, linewidth=3.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
             
                legend_names.append(model_names[i])
                shape_a=list_hlfs[i].GetECPhis()[0].shape[0]

                selected_ref = [list_hlfs[i].GetECPhis()[j].reshape(shape_a,1) for j in keys]#turning into GeV
                combined_ref = np.concatenate(selected_ref, axis=1)

                mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
                sub_label = model_names[i] if out_idx==0 else None
        
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, label=sub_label, bins=bins, color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=3., alpha=0.8, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=seps

                if(ratio):
                    eps = 1e-8
                    h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

                    ax1[out_idx//col][out_idx%col].axhline(y=0.0, color='black', linestyle='-',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=10, color='gray', linestyle='--',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=-10, color='gray', linestyle='--',linewidth=2)

                    xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
                    ax1[out_idx//col][out_idx%col].plot(xaxis,h_ratio,color=model_to_color_dict[model_names[i]],linestyle='-',linewidth = 3)
                    ax1[out_idx//col][out_idx%col].set_ylabel('Diff. (%)',fontsize=YLABEL_SIZE)
                    ax1[out_idx//col][out_idx%col].set_ylim([-50,50])


                    
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {} - {}".format(keys[0],keys[4]),fontsize=XLABEL_SIZE)
            
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5) 
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)
    plt_label = "Center of Enegery in $\\phi$ (mm) direction for Dataset "+str(dataset)
  
    fig0.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE,loc='upper center', bbox_to_anchor=[0.5, 1.06],ncol=4,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2)

    filename0 = os.path.join(output_dir, 'EC_phi_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350,bbox_inches='tight')
    
        
    if ratio:
        fig1.tight_layout()
        filename1 = os.path.join(output_dir, 'EC_phi_dataset_{}_particle_{}diff.pdf'.format( str(dataset),particle))
        fig1.savefig(filename1, dpi=300)
    
    emd_file=os.path.join(output_dir,"emd_EC_phi_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir,"separation_EC_phi_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    
    
def plot_ECPhis(list_hlfs, dataset,output_dir, particle, model_names, ratio = False,row=2,col=2,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots center of energy in phi for dataset 1 """
    EMDs={}
    Seps={}
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=False,sharey=True,squeeze=False)
   
    ax0.flatten()
    g_index=model_names.index('Geant4')
     
    reference_class=list_hlfs[g_index]
    for out_idx, key in enumerate(reference_class.GetECPhis().keys()):
    
        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-250., 250.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)

        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(reference_class.GetECPhis()[key], bins=bins,
                                                               color = model_to_color_dict[model_names[g_index]],
                                                        label=model_names[g_index], density=True, histtype='step',
                                                        alpha=1.0, linewidth=2.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(list_hlfs[i].GetECPhis()[key], label=model_names[i], bins=bins,
                                             color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=3., alpha=0.5, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(key)]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(key)]=seps
                
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {}".format(key),fontsize=XLABEL_SIZE)
            #plt.xlabel(r'[mm]')
            #ax0[out_idx//col][out_idx%col].set_xlim(*lim)
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5) 
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)

            
    plt_label = "Center of Enegery in $\\phi$ (mm) direction for Dataset "+str(dataset)
   
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig0.legend(lines[:len(model_names)], labels[:len(model_names)], loc='upper center',bbox_to_anchor=[0.55, 1.08],
                ncol = 4,fontsize=LEGEND_SIZE,borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
                handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2) #for positioning figure
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'EC_phi_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    new_path_emd=output_dir
    emd_file=os.path.join(new_path_emd,"emd_EC_phi_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    new_path_sep=output_dir
    sep_file=os.path.join(new_path_sep,"separation_EC_phi_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()

def plot_SW_etas_group(list_hlfs, dataset,output_dir,particle, model_names, ratio = False, row=3,col=3,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """  plots shower width in eta direction for dataset 2 and 3 """
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=True)

    EMDs={}
    Seps={}
    legend_names=['Geant4']
    gkeys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    if ratio:
        fig1, ax1 = plt.subplots(row,col,figsize=(width*col,height*row))
    for out_idx,keys in enumerate(gkeys):
        
        if dataset in ['2', '3']:
            lim = (0, 30.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        
        g_index=model_names.index('Geant4')
        
        reference_class=list_hlfs[g_index]

        shape_a=reference_class.GetWidthEtas()[0].shape[0]

        selected_ref = [reference_class.GetWidthEtas()[i].reshape(shape_a,1) for i in keys]#turning into GeV
        combined_ref = np.concatenate(selected_ref, axis=1)
        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
        
        main_label = model_names[g_index] if out_idx==0 else None

        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, bins=bins,color = model_to_color_dict[model_names[g_index]],
                                    label=main_label, density=True, histtype='step',
                                    alpha=1.0, linewidth=3.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                
                legend_names.append(model_names[i])
                shape_a=list_hlfs[i].GetWidthEtas()[0].shape[0]

                selected_ref = [list_hlfs[i].GetWidthEtas()[j].reshape(shape_a,1) for j in keys]#turning into GeV
                combined_ref = np.concatenate(selected_ref, axis=1)

                mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
                sub_label = model_names[i] if out_idx==0 else None
        
        
                counts_data, _, _ =  ax0[out_idx//col][out_idx%col].hist(mean_ref, label=sub_label, bins=bins, color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=2., alpha=0.8, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=seps

                if(ratio):
                    eps = 1e-8
                    h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

                    ax1[out_idx//col][out_idx%col].axhline(y=0.0, color='black', linestyle='-',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=10, color='gray', linestyle='--',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=-10, color='gray', linestyle='--',linewidth=2)

                    xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
                    ax1[out_idx//col][out_idx%col].plot(xaxis,h_ratio,color=model_to_color_dict[model_names[i]],linestyle='-',linewidth = 3)
                    ax1[out_idx//col][out_idx%col].set_ylabel('Diff. (%)',fontsize=YLABEL_SIZE)
                    ax1[out_idx//col][out_idx%col].set_ylim([-50,50])


                   
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {} - {}".format(keys[0],keys[4]),fontsize=XLABEL_SIZE)
           
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5) 
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)
    plt_label = "Shower width in $\\eta$ (mm) direction for Dataset "+str(dataset)
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
   
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE,loc='upper center', bbox_to_anchor=[0.5, 1.06],ncol=4,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2)
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'SW_eta_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    
    
    if ratio:
        fig1.tight_layout()
        filename1 = os.path.join(output_dir, 'SW_eta_dataset_{}_particle_{}diff.pdf'.format( str(dataset),particle))
        fig1.savefig(filename1, dpi=300)
    
    

    emd_file=os.path.join(output_dir,"emd_SW_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir, "separation_SW_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    
    
def plot_SW_Etas(list_hlfs, dataset,output_dir, particle,model_names, ratio = False,row=1,col=2,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots shower width in eta direction for dataset1 (photon)  """
    EMDs={}
    Seps={}
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=True)
    
    ax0.flatten()
    g_index=model_names.index('Geant4')
     
    reference_class=list_hlfs[g_index]
    for out_idx, key in enumerate(reference_class.GetWidthEtas().keys()):
        
        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (0., 100.)

        bins = np.linspace(*lim, 101)

        counts_ref, _, _ = ax0[out_idx].hist(reference_class.GetWidthEtas()[key], bins=bins,
                                                               color = model_to_color_dict[model_names[g_index]],
                                                        label=model_names[g_index], density=True, histtype='step',
                                                        alpha=1.0, linewidth=2.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                
                counts_data, _, _ = ax0[out_idx].hist(list_hlfs[i].GetWidthEtas()[key], label=model_names[i], bins=bins,
                                             color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=3., alpha=0.5, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(key)]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(key)]=seps
                
            cur_ylabel = "A.U." if out_idx==0 else ''
            ax0[out_idx].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx].set_xlabel(r"Layer {}".format(key),fontsize=XLABEL_SIZE)
            #plt.xlabel(r'[mm]')
            #ax0[out_idx//col][out_idx%col].set_xlim(*lim)
            ax0[out_idx].set_ylim([None,YMAX])
            ax0[out_idx].set_yscale('log')
            ax0[out_idx].margins(0.05, 0.5)
            ax0[out_idx].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx].tick_params(axis='y', labelsize=YTICK_SIZE)

            
    plt_label = "Shower width in $\\eta$ (mm) direction for Dataset "+str(dataset)
    fig0.suptitle(plt_label,y=1.10,fontsize=TITLE_SIZE) # for positioning title
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig0.legend(lines[:4], labels[:4], loc='upper center',bbox_to_anchor=[0.55, 1.08],
                ncol = 4,fontsize=LEGEND_SIZE,borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
                handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2) #for positioning figure
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'SW_eta_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    
    emd_file=os.path.join(output_dir,"emd_SW_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    new_path_sep=output_dir+'Sep/'
    sep_file=os.path.join(output_dir,"separation_SW_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    
    
    
    
def plot_SW_Etas_pion(list_hlfs, dataset,output_dir, particle, model_names, ratio = False,row=2,col=2,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots shower width in eta direction for dataset1 (pion)  """
    
    EMDs={}
    Seps={}
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=False,sharey=True,squeeze=False)
    
    ax0.flatten()
    g_index=model_names.index('Geant4')
  
   
    reference_class=list_hlfs[g_index]
    for out_idx, key in enumerate(reference_class.GetWidthEtas().keys()):

        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (0., 500.)
        else:
            lim = (0., 300.)

        bins = np.linspace(*lim, 101)

        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(reference_class.GetWidthEtas()[key], bins=bins,
                                                               color = model_to_color_dict[model_names[g_index]],
                                                        label=model_names[g_index], density=True, histtype='step',
                                                        alpha=1.0, linewidth=2.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                
        
        
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(list_hlfs[i].GetWidthEtas()[key], label=model_names[i], bins=bins,
                                             color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=3., alpha=0.5, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(key)]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(key)]=seps
                
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {}".format(key),fontsize=XLABEL_SIZE)
          
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5)
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)

            
   
    plt_label = " $\\eta$ (mm) direction"
   
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig0.legend(lines[:len(model_names)], labels[:4], loc='upper center',bbox_to_anchor=[0.55, 1.08],
                ncol = 4,fontsize=LEGEND_SIZE,borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
                handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2) #for positioning figure
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'SW_eta_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
   
    emd_file=os.path.join(output_dir,"emd_SW_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
   
    sep_file=os.path.join(output_dir,"separation_SW_eta_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()

def plot_sep_emd(filename, output_dir, dataset, particle,width=7,height=5,taskname='separation_power'):
    """
     plots separation power or emd score in a graph, we only show for dataset 2 and 3 in the paper.
    """
    CaloScore = {}
    CaloINN = {}
    CaloDiffusion = {}
    
    inn_counter = 0
    diffusion_counter = 0
    score_counter = 0
    
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        key, value = line.strip().split(': ')
        value = float(value)

        # Determine the type and index
        if 'CaloINN' in key:
            CaloINN[inn_counter] = math.log10(value)
            inn_counter += 1
        elif 'CaloDiffusion' in key:
            CaloDiffusion[diffusion_counter] = math.log10(value)
            diffusion_counter += 1
        elif 'CaloScore' in key:
            CaloScore[score_counter] = math.log10(value)
            score_counter += 1
            
    x_inn = list(CaloINN.keys())
    y_inn = list(CaloINN.values())
    
    
    x_diffusion = list(CaloDiffusion.keys())
    y_diffusion = list(CaloDiffusion.values())
    
    

    x_score = list(CaloScore.keys())
    y_score = list(CaloScore.values())
    
    
    
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    ax.plot(x_inn, y_inn, marker='^', label='CaloINN', color=model_to_color_dict['CaloINN'])
    ax.plot(x_diffusion, y_diffusion, marker='o', label='CaloDiffusion', color=model_to_color_dict['CaloDiffusion'])
    ax.plot(x_score, y_score, marker='*', label='CaloScore', color=model_to_color_dict['CaloScore'])

    ax.set_xlabel('Layer number',fontsize=32)
    ax.set_ylabel(f'{taskname}',fontsize=28)

    pos=np.arange(len(x_inn))
    
    xtick_label=['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44']
    
    ax.set_xticks(pos)
    ax.set_xticklabels(xtick_label)
    ax.set_ylim([-5,0])
    ax.tick_params(axis='y',labelsize=28)
    ax.tick_params(axis='x',rotation=90,labelsize=28)
    
    
    ax.legend(loc='best',fontsize=30,borderpad=0.05,labelspacing=0.05,handlelength=0.5,
         handleheight=0.25,handletextpad=0.1,borderaxespad=0.1,columnspacing=0.1)
    #plt.grid(True)
    fig.tight_layout()
    filename=taskname+f'_dataset_{dataset}_particle_{particle}.pdf'
    filename=os.path.join(output_dir,filename)
    fig.savefig(filename)
   
    
    
def plot_SW_Phis(list_hlfs, dataset,output_dir, particle, model_names, ratio = False,row=2,col=2,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    """ plots shower width in phi direction for dataset 1   """
    EMDs={}
    Seps={}
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=False,sharey=True,squeeze=False)
  
    ax0.flatten()
    g_index=model_names.index('Geant4')
        
   
    reference_class=list_hlfs[g_index]
    for out_idx, key in enumerate(reference_class.GetWidthPhis().keys()):
        
        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (0., 500.)
        else:
            lim = (0., 300.)

        bins = np.linspace(*lim, 101)

        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(reference_class.GetWidthPhis()[key], bins=bins,
                                                               color = model_to_color_dict[model_names[g_index]],
                                                        label=model_names[g_index], density=True, histtype='step',
                                                        alpha=1.0, linewidth=2.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                
        
        
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(list_hlfs[i].GetWidthPhis()[key], label=model_names[i], bins=bins,
                                             color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=3., alpha=0.5, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(key)]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(key)]=seps
                
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {}".format(key),fontsize=XLABEL_SIZE)
           
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5)
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)

            
    
    plt_label = "$\\phi$ (mm) direction "
   
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig0.legend(lines[:len(model_names)], labels[:len(model_names)], loc='upper center',bbox_to_anchor=[0.55, 1.08],
                ncol = 4,fontsize=LEGEND_SIZE,borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
                handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2) #for positioning figure
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'SW_phi_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    new_path_emd=output_dir
    emd_file=os.path.join(new_path_emd,"emd_SW_phi_dataset_{}_particle_{}.txt".format(str(dataset),particle))
    write_dict_to_txt(EMDs,emd_file)
    new_path_sep=output_dir
    sep_file=os.path.join(new_path_sep,"separation_SW_phi_dataset_{}_particle_{}.txt".format((dataset),particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    

def plot_SW_Phis_group(list_hlfs, dataset,output_dir, particle,model_names, ratio = False,row=3,col=3,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    
    """ plots shower width in phi direction for dataset 2 and 3 """
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=True)

    EMDs={}
    Seps={}
    legend_names=['Geant4']
    gkeys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    dataset=str(dataset)
    if ratio:
        fig1, ax1 = plt.subplots(row,col,figsize=(width*col,height*row))
    for out_idx,keys in enumerate(gkeys):
       


        if dataset in ['2', '3']:
            lim = (0, 30.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        
        g_index=model_names.index('Geant4')
       
        reference_class=list_hlfs[g_index]

        shape_a=reference_class.GetWidthPhis()[0].shape[0]

        selected_ref = [reference_class.GetWidthPhis()[i].reshape(shape_a,1) for i in keys]#turning into GeV
        combined_ref = np.concatenate(selected_ref, axis=1)
        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
        main_label = model_names[g_index] if out_idx==0 else None

        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, bins=bins,color = model_to_color_dict[model_names[g_index]],
                                    label=main_label, density=True, histtype='step',
                                    alpha=1.0, linewidth=3.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                legend_names.append(model_names[i])
               
                shape_a=list_hlfs[i].GetWidthPhis()[0].shape[0]

                selected_ref = [list_hlfs[i].GetWidthPhis()[j].reshape(shape_a,1) for j in keys]#turning into GeV
                combined_ref = np.concatenate(selected_ref, axis=1)

                mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
                sub_label = model_names[i] if out_idx==0 else None
        
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, label=sub_label, bins=bins, color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=2., alpha=0.8, density=True)
            
                emd_score=getEMD(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=emd_score

                seps = _separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=seps

                if(ratio):
                    eps = 1e-8
                    h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

                    ax1[out_idx//col][out_idx%col].axhline(y=0.0, color='black', linestyle='-',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=10, color='gray', linestyle='--',linewidth=2)
                    ax1[out_idx//col][out_idx%col].axhline(y=-10, color='gray', linestyle='--',linewidth=2)

                    xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
                    ax1[out_idx//col][out_idx%col].plot(xaxis,h_ratio,color=model_to_color_dict[model_names[i]],linestyle='-',linewidth = 3)
                    ax1[out_idx//col][out_idx%col].set_ylabel('Diff. (%)',fontsize=YLABEL_SIZE)
                    ax1[out_idx//col][out_idx%col].set_ylim([-50,50])


                    

            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel,fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {} - {}".format(keys[0],keys[4]),fontsize=XLABEL_SIZE)
            
            ax0[out_idx//col][out_idx%col].set_ylim([None,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5)
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)
            
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig0.legend(lines, labels, loc='upper center',ncol = 2,fontsize=LEGEND_SIZE,bbox_to_anchor=[0.5, 1.18]) #for positioning figure
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE,loc='upper center', bbox_to_anchor=[0.5, 1.06],ncol=4,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2)
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'SW_phi_dataset_{}_particle_{}.pdf'.format( dataset,particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    
    
    if ratio:
        fig1.tight_layout()
        filename1 = os.path.join(output_dir, 'SW_phi_dataset_{}_Partucle_{}_diff.pdf'.format( dataset,particle))
        fig1.savefig(filename1, dpi=300)
    
    emd_file=os.path.join(output_dir,"emd_SW_phi_dataset_{}_particle_{}.txt".format(dataset,particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir,"separation_SW_phi_dataset_{}_particle_{}.txt".format(dataset,particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    

