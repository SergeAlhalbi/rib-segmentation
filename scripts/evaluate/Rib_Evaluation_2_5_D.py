# -*- coding: utf-8 -*-
"""

Created on Wed Apr 19 13:40:18 2023

@author: matlab4

2.5D UNet Evaluation
Riverain Tech 2023

"""

import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.io import savemat

# ----------------------------------------------------------------------------

# Evaluation functions
def Dice_Per_Volume(true_mask, pred_mask):
    """
    Computes the dice coefficients per volume between two binary masks volumes.
    
    Inputs:
        true_mask (numpy array): binary array of shape (depth, height, width) representing the ground truth mask.
        pred_mask (numpy array): binary array of shape (depth, height, width) representing the predicted mask.
    
    Returns:
        dice_volume (float): the overall dice score for the entire volume.
    """
    
    # Convert type
    true_mask = true_mask.astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)
    
    if (pred_mask.shape) != (true_mask.shape):
        raise ValueError('Unmatched masks sizes')
    
    intersection = np.sum(np.logical_and(pred_mask, true_mask))
    denominator = np.sum(pred_mask) + np.sum(true_mask)
        
    if denominator == 0:
        dice_volume = 0
    else:
        dice_volume = 2 * intersection / denominator
        
    return dice_volume


def Dice_Per_Slices(true_mask, pred_mask):
    """
    Computes the dice coefficients per slices between two binary masks volumes.
    
    Inputs:
        true_mask (numpy array): Binary array of shape (height, width, depth) representing the ground truth mask
        pred_mask (numpy array): Binary array of shape (height, width, depth) representing the predicted mask
    
    Returns:
        dice_per_slice (numpy array - float32): A 1xN vector of dice scores for the entire volume, where N is the number of slices
    """
    
    # Convert type
    true_mask = true_mask.astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)
    
    if (pred_mask.shape)[-1] != (true_mask.shape)[-1]:
        raise ValueError('Unmatched masks sizes')
        
    slices_n = (true_mask.shape)[-1]
    dice_per_slice = np.zeros(slices_n, dtype=np.float32)

    # Loop over slices
    for slice_n in range(slices_n):
        pred_mask_slice = pred_mask[:,:,slice_n]
        true_mask_slice = true_mask[:,:,slice_n]
        
        intersection = np.sum(np.logical_and(pred_mask_slice, true_mask_slice))
        denominator = np.sum(pred_mask_slice) + np.sum(true_mask_slice)
        
        if denominator == 0:
            dice_per_slice[slice_n] = 0
        else:
            dice_per_slice[slice_n] = 2 * intersection / denominator
            
    return dice_per_slice

# ----------------------------------------------------------------------------

# Main
if __name__ == '__main__':
    
    # Saving path
    main_save_path = r'D:\Serge 231\Rib\Analysis'
    
    # Load grouping list
    list_path = r'D:\Serge 231\Rib\Data\Rib_Lists\grpsList'
    list_grps_dict = sio.loadmat(list_path)
    list_grps = list_grps_dict['grpsList']
    
    file_ns = len(list_grps)
    
    # Initialize object
    patient_dice_list = np.empty((4, file_ns), dtype=object)
        
    # Loop over files
    for file_n in range(file_ns):
        
        # Get the folder name
        folder_nm = list_grps[file_n,0][0].split('\\')[0]
        
        # Get the patient number
        patient_n = list_grps[file_n,0][0].split('\\')[1]
        
        # Get the group number
        grp_n = list_grps[file_n,1][0][0]
        
        # Directories
        true_path = os.path.join('D:', '\Serge 231', 'Rib', 'Data', 'Rib_Inference', folder_nm, 'Group' + str(grp_n), 'pos', patient_n + '.mat')
        pred_path = os.path.join('D:', '\Serge 231', 'Rib', 'Data', 'Rib_Inference_2_5_D', folder_nm, 'Group' + str(grp_n), 'pos', patient_n + '.mat')
        
        # Load
        true_dict = sio.loadmat(true_path)
        pred_dict = sio.loadmat(pred_path)
         
        # Extract
        true_mask = true_dict['predictedRibMask']
        pred_mask = pred_dict['predictedRibMask_2_5_D']
        
        # Compute
        dice_volume = Dice_Per_Volume(true_mask, pred_mask)
        
        # Status update
        print(f"({file_n+1}/{file_ns}) Dice coefficient for patient {patient_n} in {folder_nm} - Group{grp_n} is: {dice_volume}")
        
        # Update object
        patient_dice_list[0,file_n] = folder_nm
        patient_dice_list[1,file_n] = grp_n
        patient_dice_list[2,file_n] = patient_n
        patient_dice_list[3,file_n] = dice_volume
    
    # Sort from low to high
    sorted_patient_indices = np.argsort(patient_dice_list[3])
    sorted_patient_dice_list = patient_dice_list[:, sorted_patient_indices]
    
    # Save objects
    df = pd.DataFrame(patient_dice_list.T, columns=['Dataset', 'Group', 'Patient', 'Dice'])
    df.to_excel(os.path.join(main_save_path, 'Rib_Metrics', '2_5_D') + '\dice_list.xlsx', index=False)
    df_s = pd.DataFrame(sorted_patient_dice_list.T, columns=['Dataset', 'Group', 'Patient', 'Dice'])
    df_s.to_excel(os.path.join(main_save_path, 'Rib_Metrics', '2_5_D') + '\sorted_dice_list.xlsx', index=False)
    print('Lists have been saved')
    
    # Extract the first cases with lowest dice scores for review
    first_lowest = file_ns
    
    for file_n_r in range(first_lowest):
        
        # Get the folder name
        folder_nm_r = sorted_patient_dice_list[0,file_n_r]
        
        # Get the patient number
        patient_n_r = sorted_patient_dice_list[2,file_n_r]
        
        # Get the group number
        grp_n_r = sorted_patient_dice_list[1,file_n_r]
        
        # Status update
        print(f"({file_n_r+1}/{first_lowest}) Processing dices per slices for patient {patient_n_r} in {folder_nm_r} - Group{grp_n_r}")
        
        # Directories
        true_path_r = os.path.join('D:', '\Serge 231', 'Rib', 'Data', 'Rib_Inference', folder_nm, 'Group' + str(grp_n), 'pos', patient_n + '.mat')
        pred_path_r = os.path.join('D:', '\Serge 231', 'Rib', 'Data', 'Rib_Inference_2_5_D', folder_nm, 'Group' + str(grp_n), 'pos', patient_n + '.mat')
        
        # Load
        true_dict_r = sio.loadmat(true_path_r)
        pred_dict_r = sio.loadmat(pred_path_r)
         
        # Extract
        true_mask_r = true_dict_r['predictedRibMask']
        pred_mask_r = pred_dict_r['predictedRibMask_2_5_D']
        
        # Compute dice
        dice_slice_r = Dice_Per_Slices(true_mask_r, pred_mask_r)
        
        # Sort
        sorted_slice_indices = np.argsort(dice_slice_r)
        sorted_dice_per_slice_r = dice_slice_r[sorted_slice_indices]
        
        # Store
        slice_dice = [sorted_slice_indices + 1, sorted_dice_per_slice_r]
        
        # Save
        slice_dice_dict = {"sliceDice": slice_dice}
        savemat(os.path.join(main_save_path, 'Rib_Review', '2_5_D', str(file_n_r + 1) + '_' + folder_nm_r + '_' + patient_n_r + '.mat'), slice_dice_dict)
        
    print('Files have been saved')
    print('Done!')

# ----------------------------------------------------------------------------