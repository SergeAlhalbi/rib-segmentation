# -*- coding: utf-8 -*-
"""

Created on Thu Mar 30 16:44:16 2023

@author: matlab4

2D UNet Inference
Riverain Tech 2023

"""

import tensorflow as tf
from tensorflow.keras.models import load_model

import os
import glob
import scipy.io as sio
from scipy.io import savemat
import numpy as np

# ----------------------------------------------------------------------------

# Preprocessing function
def process_mat_inference(im_input, crop_bool=False, crop_sz=[96, 96],
                          window_rng_bool=False, window_rng_default=[-1150, 350],
                          window_rng_type='add-div',
                          ):
    '''
    Preprocesses an image for inference by center cropping and applying Hounsfield unit (HU) windowing.
    
    Inputs:
        im_input (numpy array): Input image of shape [height, width]
        crop_bool (bool): Flag indicating whether to perform center cropping or not
        crop_sz (list of ints): Size of the center crop, specified as [height, width]
        window_rng_bool (bool): Flag indicating whether to apply HU range windowing or not
        window_rng_default (list of ints): Default range of HU values to window, specified as [lower, upper]
        window_rng_type (str): Type of HU range windowing to apply, options include 'add-div', 'norm-clip', and 'norm'
    
    Returns:
        im_input (tensorflow tensor - float32): Preprocessed image tensor of shape [height, width, channels]
    '''
    
    # Extract info
    sz = im_input.shape
    transX = 0
    transY = 0
    rng = window_rng_default
    
    # Center crop wwith translation
    if crop_bool:
        
        x1 = transX + sz[0]//2 - crop_sz[0]//2
        x2 = transX + crop_sz[0] + sz[0]//2 - crop_sz[0]//2
        y1 = transY + sz[1]//2 - crop_sz[1]//2
        y2 = transY + crop_sz[1] + sz[0]//2 - crop_sz[1]//2
        im_input = im_input[ x1 : x2, y1 : y2]

    # HU range window
    if window_rng_bool:
        
        if (window_rng_type=='add-div'):
            im_input = im_input + rng[0]
            im_input = im_input / rng[1]
            
        elif (window_rng_type=='norm-clip'):
            im_input = im_input - rng[0]
            im_input = im_input / (rng[1]-rng[0])
            im_input = tf.maximum(0.0, tf.minimum(1.0, im_input))
    
        elif (window_rng_type=='norm'):
            im_input = im_input - tf.reduce_min(im_input)
            im_input = im_input / tf.reduce_max(im_input)
            
        else:
            raise ValueError('Supported window_rng_types are add-div, norm-clip, or norm')
    
    # Add dimension if necessary
    if len(im_input.shape) == 2:
        im_input = im_input[:, :, np.newaxis]
    
    # Cast explicitly
    im_input = tf.cast(im_input, tf.float32)
    
    return im_input

# ----------------------------------------------------------------------------

# Main
if __name__ == '__main__':
    
    # Data and saving paths
    data_path = r'D:\Serge 231\Rib\Data\Rib_Volumes'
    main_save_path = r'D:\Serge 231\Rib\Data\Rib_Inference'
    
    # Load grouping list
    list_path = r'D:\Serge 231\Rib\Data\Rib_Lists\grpsList'
    list_grps_dict = sio.loadmat(list_path)
    list_grps = list_grps_dict['grpsList']
    
    # Load model
    model_folder_path = r'D:\Serge 231\Rib\Models\Rib_Models\20230410_1025--2-d4f16s512b16-20230410_512_2_RibSeg_Group1'
    model_path = glob.glob(model_folder_path+'\\*.onnx')
    model_path = [model_path[0][:-4]+'h5']
    model = load_model(model_path[0], custom_objects={'CustomMetric_Dice': tf.keras.metrics.Accuracy})
    
    file_ns = len(list_grps)
    # Loop over files
    for file_n in range(file_ns):
        
        # Get the folder name
        folder_nm = list_grps[file_n,0][0].split('\\')[0]
        
        # Get the patient number
        patient_n = list_grps[file_n,0][0].split('\\')[1]
        
        # Get the group number
        grp_n = list_grps[file_n,1][0][0]
        
        # Status update
        print(f"({file_n+1}/{file_ns}) Processing patient {patient_n} in {folder_nm} - Group{grp_n}")
        
        # Get the file path
        vol_path = os.path.join(data_path, folder_nm, patient_n + '.mat')
            
        # Load
        vol_dict = sio.loadmat(vol_path)
        
        # Extract
        vol = vol_dict['volume']
        
        # Directory
        save_path = os.path.join(main_save_path, folder_nm, 'Group' + str(grp_n), 'pos', patient_n + '.mat')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
    
        # Loop over slices
        mask_vol = np.empty((vol.shape[0], vol.shape[1], vol.shape[2]), dtype=np.uint8)
        for im_nb in range(vol.shape[2]):
            
            #  Extract
            im_input = vol[:,:,im_nb]
            
            # Preprocess
            im = process_mat_inference(im_input, crop_bool=False,
                                       window_rng_bool=True,
                                       window_rng_default=[0, 4095],
                                       window_rng_type='norm-clip')
            model_input = im[np.newaxis, ...]
            
            # Predict
            one_hot_out = np.squeeze(model.predict(model_input))
            
            # Threshold
            mask = tf.cast(tf.argmax(one_hot_out, axis=2, output_type=tf.int32), tf.uint8)
            
            # Store
            mask_array = mask.numpy()
            mask_vol[:,:,im_nb] = mask_array
            
        # Save
        mask_dict = {"predictedRibMask": mask_vol}
        savemat(save_path, mask_dict)
            
    print('Files have been saved')
    print('Done!')
        
# ----------------------------------------------------------------------------