# -*- coding: utf-8 -*-
"""

Created on Thu Feb  2 11:15:02 2023

@author: matlab7

2D UNet Script
Dennis Wu
Riverain Tech 2023

"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
    Dropout, MaxPooling2D, Conv2DTranspose, concatenate, ReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import glob
import random
import scipy.io as sio
import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tf2onnx
import onnx

NAME_INPUT = 'im'
NAME_TARGET = 'msk'

gpu_to_use = 0  #0 or 1 for 109

# GPU set-up
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_to_use], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_to_use], True)
        # # Currently, memory growth needs to be the same across GPUs
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # logical_gpus = tf.config.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ----------------------------------------------------------------------------
# Custom Metrics
# ----------------------------------------------------------------------------
class CustomMetric_Dice(tf.keras.metrics.Metric):
    
    def __init__(self, class_axis=1, **kwargs):
        
        name="custom_dice" + str(class_axis)
        
        super(CustomMetric_Dice, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name="dice", initializer="zeros")
        
        self.count = self.add_weight(name="count", initializer="zeros")
        
        self.axis = class_axis       
            
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_pred_f32 = tf.cast( tf.argmax(y_pred, axis=-1), tf.float32)
        y_true_f32 = tf.cast( tf.argmax(y_true, axis=-1), tf.float32)

        # Get only predictions of class axis  
        y_pred_bool = y_pred_f32 == self.axis
        y_true_bool = y_true_f32 == self.axis

        # Compute dice only if non-bkg class exists
        if tf.math.reduce_any( y_true_bool ):
            intersection = tf.logical_and(y_pred_bool, y_true_bool)
            sum_pred = tf.reduce_sum( tf.cast(y_pred_bool, tf.float32) )
            sum_true = tf.reduce_sum( tf.cast(y_true_bool, tf.float32) )
            self.dice.assign_add( 2 *  tf.reduce_sum( tf.cast(intersection, tf.float32) ) / \
                                 ( sum_pred + sum_true ) )
            self.count.assign_add(1)

    def result(self):
        if self.count > 0:
            return self.dice/self.count
        else:
            return 0.

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dice.assign(0)
        self.count.assign(0)

# ----------------------------------------------------------------------------
# Build U-Net (2D)
# ----------------------------------------------------------------------------

def unet2D(sz, depth=3, filts=16, num_classes=2, drop_rate=0.5, 
            batch_norm_bool=False, final_bias=0.5, final_activation='softmax'):
    print('sz =', sz)
    print('channel dimension has length = ', sz[2])
    print('depth =', depth)
    print('filts =', filts)
    print('drop rate =', drop_rate)
    print('batch norm =', batch_norm_bool)
    
    # Build U-Net 2D
    inputs = Input(shape = (sz[0],sz[1],sz[2]), 
                   name='Input') # sz[2] is channel
    
    net = inputs
    c = []    
    last_ax = len(sz)
    print('last axis =', last_ax)
    
    # Encoder layers
    for i in range(depth):  #FYI, for some, batchnorm in-between conv and relu (paper)
        net = Conv2D(filts*2**i, 3, kernel_initializer='he_normal', padding='same', 
                     name='Conv2D_Enc_'+str(i)+'_1') (net)
        if batch_norm_bool:
            net = BatchNormalization(axis=last_ax,
                                     name='BatchNorm_Enc_'+str(i)+'_1') (net)
        net = ReLU() (net)
        
        net = Conv2D(filts*2**i, 3, kernel_initializer='he_normal', padding='same',
                     name='Conv2D_Enc_'+str(i)+'_2') (net)
        if batch_norm_bool:
            net = BatchNormalization(axis=last_ax) (net)     
        net = ReLU() (net)
        
        c.append(net)
        net = MaxPooling2D((2,2), padding='same') (c[i]) #<-added padding same for maxpool
        
        
    # Middle layers
    net = Dropout(drop_rate) (net) 
    
    i = depth;
    net = Conv2D(filts*2**i, 3, kernel_initializer='he_normal', padding='same',
                 name='Conv2D_Bridge_'+str(i)+'_1') (net)
    if batch_norm_bool:
        net = BatchNormalization(axis=last_ax) (net)
    net = ReLU() (net)
    
    net = Conv2D(filts*2**i, 3, kernel_initializer='he_normal', padding='same',
                 name='Conv2D_Bridge_'+str(i)+'_2') (net)
    if batch_norm_bool:
        net = BatchNormalization(axis=last_ax) (net)
    net = ReLU() (net)
    
    
    # Decoder layers
    for i in reversed(range(depth)):
        net = Conv2DTranspose(filts*2**i, 3, strides=(2, 2), padding='same',
                              name='Conv2DTrans_'+str(i)) (net)
        net = concatenate([net, c[i]], axis=last_ax)
        
        net = Conv2D(filts*2**i, 3, kernel_initializer='he_normal', padding='same',
                     name='Conv2D_Dec_'+str(i)+'_1') (net)
        if batch_norm_bool:
            net = BatchNormalization(axis=last_ax)(net)
        net = ReLU() (net)
        
        net = Conv2D(filts*2**i, 3, kernel_initializer='he_normal', padding='same',
                     name='Conv2D_Dec_'+str(i)+'_2') (net)
        if batch_norm_bool:
            net = BatchNormalization(axis=last_ax)(net)
        net = ReLU() (net)
        
    initializer = tf.keras.initializers.Constant(final_bias)
    outputs = Conv2D(num_classes, 1, activation=final_activation, 
                     bias_initializer=initializer, padding='same',
                     name='Conv2D_Final') (net) #<-adding padding same explicitly
    
    
    # Build model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary(120)    

    return model

# ----------------------------------------------------------------------------
# Process mat files
# ----------------------------------------------------------------------------

def build_arg_list( crop_sz=None, 
                    border_clear_sz=None, 
                    window_rng_default=None, window_rng_type=None,
                    one_hot_bool=False, num_classes=2, 
                    row_flip_prob=0.0, col_flip_prob=0.0, z_flip_prob=0.0, 
                    trans_rng=None, ang_rng=None,
                    level_rng=None, window_rng=None,
                    blur_prob=0.0, blur_sigma_rng=[0.5,0.7],
                    sharp_prob=0.0, sharp_sigma_rng=[0.5,0.9],
                    noise_prob=0.0, noise_perc=0.01
                    ):
    
    
    crop_bool = crop_sz is not None
    if not crop_bool:
        crop_sz=''
        
    border_clear_bool = border_clear_sz is not None
    if not border_clear_bool:
        border_clear_sz=''
        
    window_rng_bool = window_rng_type is not None
    if not window_rng_bool:
        window_rng_default=''
        window_rng_type=''
        
    trans_bool = trans_rng is not None
    if not trans_bool:
        trans_rng=''
        
    rotate_bool = ang_rng is not None
    if not rotate_bool:
        ang_rng=''
        
    level_bool = level_rng is not None
    if not level_bool:
        level_rng=''
        
    window_bool = window_rng is not None
    if not window_bool:
        window_rng=''
    
    # num classes needs to be first
    arg_list = [    crop_bool, crop_sz, 
                    border_clear_bool, border_clear_sz, 
                    window_rng_bool, window_rng_default, window_rng_type,
                    one_hot_bool, num_classes, 
                    row_flip_prob, col_flip_prob, z_flip_prob, 
                    trans_bool, trans_rng, 
                    rotate_bool, ang_rng,
                    level_bool, level_rng, 
                    window_bool, window_rng,
                    blur_prob, blur_sigma_rng,
                    sharp_prob, sharp_sigma_rng,
                    noise_prob, noise_perc]
    
    
    return arg_list



def process_mat_train(input_name, aug_bool=True,
                      crop_bool=False, crop_sz=[96, 96],
                      border_clear_bool=False, border_clear_sz=5,
                      window_rng_bool=False, window_rng_default=[-1150, 350], window_rng_type='add-div',
                      one_hot_bool=False, num_classes=2,
                      row_flip_prob=0.5, col_flip_prob=0.5, z_flip_prob=0.5,
                      trans_bool=False, trans_rng=[-5, 5],
                      rotate_bool=False, ang_rng=[-15, 15],
                      level_bool=False, level_rng=[-500, -300],
                      window_bool=False, window_rng=[800, 1600],
                      blur_prob=0.05, blur_sigma_rng=[0.5,0.7],
                      sharp_prob=0.05, sharp_sigma_rng=[0.5,0.9],
                      noise_prob=0.025, noise_perc=0.01
                      ):
    
    input_name = input_name if type(input_name) == str else input_name.numpy().decode('utf8')
    
    im = sio.loadmat(input_name)
    
    im_input = im[NAME_INPUT]
    im_target = im[NAME_TARGET]
    
    transX = 0
    transY = 0
    rng = window_rng_default
    window = rng[1]-rng[0]

    # Augmentation
    if aug_bool:
            
        # row flip
        if tf.random.uniform((1,1)) < row_flip_prob:
            im_input = np.flip(im_input, axis=0)
            im_target = np.flip(im_target, axis=0)
        
        # col flip
        if tf.random.uniform((1,1)) < col_flip_prob:
            im_input = np.flip(im_input, axis=1)
            im_target = np.flip(im_target, axis=1)
            
        # z flip  
        if tf.random.uniform((1,1)) < z_flip_prob:
            im_input = np.flip(im_input, axis=2)
            im_target = np.flip(im_target, axis=2)
    
        # Translation
        if trans_bool:
            transX = tf.random.uniform((1,1), trans_rng[0], trans_rng[1], 'int32')
            transY = tf.random.uniform((1,1), trans_rng[0], trans_rng[1], 'int32')
        
        # Rotate
        if rotate_bool:
            ang = tf.random.uniform((1,1), tf.cast(ang_rng[0], tf.float32), tf.cast(ang_rng[1], tf.float32), dtype=tf.float32)
            im_input = nd.rotate(im_input, ang[0][0], mode='nearest', reshape=False)
            im_target = nd.rotate(im_target, ang[0][0], mode='nearest', reshape=False)
        
        # Window Range
        if window_bool:
            level = tf.random.uniform((1,1), level_rng[0], level_rng[1], 'int32')
            window = tf.random.uniform((1,1), window_rng[0], window_rng[1], 'int32')
            rng = [level-window//2, level+window//2]
            rng = tf.cast(rng, 'float32')
        
        
        # Image appearance
        if tf.random.uniform((1,1)) < blur_prob:
            # Blur
            sigma = blur_sigma_rng[0] + tf.random.uniform((1,1))*(blur_sigma_rng[1]-blur_sigma_rng[0])
            sigma = sigma[0][0].numpy()
            im_input = nd.gaussian_filter(im_input, sigma)
        elif tf.random.uniform((1,1)) < sharp_prob:
            # Sharpen
            sigma = sharp_sigma_rng[0] + tf.random.uniform((1,1))*(sharp_sigma_rng[1]-sharp_sigma_rng[0])
            sigma = sigma[0][0].numpy()
            im_sharp = im_input - nd.gaussian_filter(im_input, sigma)
            im_input = im_input + im_sharp
            
        # Noise    
        if np.random.uniform() < noise_prob:
            im_input = im_input + tf.random.uniform(im_input.shape, -1, 1) * tf.cast(window, 'float32') * noise_perc
            
                

    # Preprocessing
    sz = im_input.shape
    
    # Center Crop w/ trans
    if crop_bool:    
        
        x1 = transX + sz[0]//2 - crop_sz[0]//2
        x2 = transX + crop_sz[0] + sz[0]//2 - crop_sz[0]//2
        
        y1 = transY + sz[1]//2 - crop_sz[1]//2
        y2 = transY + crop_sz[1] + sz[0]//2 - crop_sz[1]//2
        
        im_input = im_input[ x1 : x2, y1 : y2]            
        im_target = im_target[ x1 : x2 , y1 : y2]
    
    
    # Throw out mask objects at the boundaries
    if border_clear_bool:
        # Create mask and get objects
        im_mask, nr_objects = nd.label(im_target)
        
        for obj in range(1, nr_objects+1):
            
            xs, ys = np.where(im_mask == obj)
            
            # if touches border
            if any(xs == 0) or any(ys == 0) or any(xs == sz[0]) or any(ys == sz[1]):
                
                center_mask = im_mask[border_clear_sz:-border_clear_sz][border_clear_sz:-border_clear_sz]
                
                # if doesn't encroach the center
                if ~np.any( center_mask == obj ):
                    
                    # remove object
                    im_target[im_mask == obj] = 0    


    # HU Range Window
    if window_rng_bool:
        if (window_rng_type=='add-div'):
            im_input = im_input + rng[0]
            im_input = im_input / rng[1]
            im_target = im_target + rng[0]
            im_target = im_target / rng[1]
            
        elif (window_rng_type=='norm-clip'):
            im_input = im_input - rng[0]
            im_input = im_input / (rng[1]-rng[0])
            im_input = tf.maximum(0.0, tf.minimum(1.0, im_input))
    
        elif (window_rng_type=='norm'):
            im_input = im_input - tf.reduce_min(im_input)
            im_input = im_input / tf.reduce_max(im_input)
            
        else:
            raise ValueError('Supported window_rng_types are add-div, norm-clip, or norm')  
    
    # One-hot encode
    if one_hot_bool:
        im_target = tf.one_hot(im_target, num_classes)
    
    
    # Add dimension if necessary
    if len(im_input.shape) == 2:
        im_input = im_input[:, :, np.newaxis]
    
    if len(im_target.shape) == 2:
        im_target = im_target[:, :, np.newaxis]
    
    # cast explicit
    im_input = tf.cast(im_input, tf.float32)
    im_target = tf.cast(im_target, tf.float32)
    
    return im_input, im_target

# ----------------------------------------------------------------------------
# View Train and validation Examples
# ----------------------------------------------------------------------------
def viewTrainVal(train_ds, val_ds):
    print('Evaluating training and validation datasets')
    
    for ds in train_ds.take(1):
        print('Input training batch shape', ds[0].shape, ds[0].dtype)
        print('Output training batch shape', ds[1].shape, ds[1].dtype)
        print(np.unique(ds[0]))
        print(np.unique(ds[1]))
        
        if len(ds) > 2:
            print('Class training weights', ds[2].shape, ds[2].dtype)            
            print(np.unique(ds[2]))

    cur_z = round(ds[0].shape[-1]/2)-1
    cur_im = np.squeeze(ds[0][0,:,:,cur_z])
    
    cur_z2 = round(ds[1].shape[-1]/2)
    cur_tar = np.squeeze(ds[1][0,:,:,cur_z2])
    
    plt.subplot(2, 2, 1)
    plt.imshow(cur_im)
    plt.title('Training set')
    plt.subplot(2, 2, 2)
    plt.imshow(cur_tar)
    plt.title('Target')

    for ds in val_ds.take(1):
        print('Input val batch shape',  ds[0].shape, ds[0].dtype)
        print('Output val batch shape', ds[1].shape, ds[1].dtype)
        print(np.unique(ds[0]))
        print(np.unique(ds[1]))
        
        if len(ds) > 2:            
            print('Class val weights', ds[2].shape, ds[2].dtype)
            print(np.unique(ds[2]))
    
    cur_z = round(ds[0].shape[-1]/2)-1
    cur_im = np.squeeze(ds[0][0,:,:,cur_z])  
    cur_z2 = round(ds[1].shape[-1]/2)
    cur_tar = np.squeeze(ds[1][0,:,:,cur_z2])
    
    plt.subplot(2, 2, 3)
    plt.imshow(cur_im)
    plt.title('Val set')
    plt.subplot(2, 2, 4)
    plt.imshow(cur_tar)
    plt.title('Target')

    plt.show(block=False)
    
    
    return


# ----------------------------------------------------------------------------
# Display Loss Graph
# ----------------------------------------------------------------------------
def viewLoss(results, timing=None):
    
    plt.figure()
    if len(results) > 2:
        axP1 = plt.subplot(1,2,1)
        axP2 = plt.subplot(1,2,2)
    else:
        axP1 = plt.subplot(1,1,1)
        axP2 = axP1
        
    lossIdx = [0, len(results)/2]    
    for idx, err in enumerate(results):
        if any(a == idx for a in lossIdx):
            plt.axes(axP1)
            plt.plot(np.log(results[err]), label=err)
            
            plt.xlabel('Epochs')
            plt.ylabel('Log loss')
            plt.legend()

        else:
            plt.axes(axP2)
            plt.plot(results[err], label=err)

            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.legend()

    
    plt.axes(axP1)
    title_str = 'Lowest val: {:.4E}'.format( min(results['val_loss']) )
    if timing is not None:
        title_str = title_str + ' | time: {0:.1f}s'.format( timing['elapsed_secs'] )
    
    plt.title(title_str)
    plt.show(block=False)
    return


# ----------------------------------------------------------------------------
# View Test Images
# ----------------------------------------------------------------------------
def viewTest(model, test_files, arg_list, cur_i=None, cur_z=None, miscTxt=None):
    print('Test dataset n =',len(test_files)) 
    
    if cur_i is None:
        cur_i = np.random.randint(0,len(test_files)-1)
    print('Displaying',cur_i)

    [cur_input_vol, cur_target_vol] = process_mat_train(test_files[cur_i], True, *arg_list)     
    print('Input shape:',cur_input_vol.shape)
    print('Input type:',cur_input_vol.dtype)
    print('Input values:',np.unique(cur_input_vol))

    sz = cur_input_vol.shape
    if cur_z is None:
        cur_z = round(sz[2]/2)-1
        
    model_input = cur_input_vol[np.newaxis, ...]  
        
    class_axis = 1
    
    # predict
    cur_pred = np.squeeze(model.predict(model_input))
    if len(cur_pred.shape) > 2:
        cur_pred = cur_pred[:,:,class_axis] # look at disease class only
    
    # cur_pred = np.argmax(cur_pred, axis=-1) # assign prediction

    print('Predicted shape:',cur_pred.shape)
    print('Predicted type:',cur_pred.dtype)
    print('Predicted values:',np.unique(cur_pred))
    
    # Display
    cur_im = cur_input_vol[:, :,cur_z]
    if len(cur_target_vol.shape) > 2 and cur_target_vol.shape[2] > 1:
        cur_tar = cur_target_vol[:,:,class_axis] # look at disease class only
    else:
        cur_tar = cur_target_vol
    
    # Display
    plt.figure()
    plot_imgs = [cur_im, cur_pred, cur_tar]
    titles = ['Input', 'Pred', 'Targ']
    xlabels = [os.path.basename(test_files[cur_i]), 'z-slice: ' + str(cur_z), miscTxt]
    
    for sp in range(3):
        plt.subplot(1, 3, sp+1)
        plt.imshow(plot_imgs[sp], vmin=0, vmax=1)
        plt.title(titles[sp])
        plt.xlabel(xlabels[sp])
    
    plt.show(block=False)

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def add_sample_weights(image, label, weights):
       
        # for one-hot:
        sample_classes = tf.argmax(label, axis=3)
        sample_weights = tf.gather(weights, indices=sample_classes)
        
        return image, label, sample_weights
    
    
def train(model, train_files, val_files, out_folder, epoch_n, batch_sz, 
          addiFmt='', 
          monitoring = 'val_loss',
          monitor_mode = 'min', 
          patience = 15,
          class_weights=None, 
          arg_list=[], ask = 0):

    # Begin timing
    start_time = time.time() 
    start_time_str = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime(start_time))
    #-----------------------------------------------------------------------------
   
    
    # Training set    
    # get filenames instead of loading everything into memory for shuffling
    def rand_gen(files):
        print('Shuffling dataset')
        num = len(files)
        inds = list(range(num))
        random.shuffle(inds)
        for a in inds:
            yield files[a]
    
    train_num = len(train_files)
    print('Train dataset n =', train_num) 
    
    train_ds = tf.data.Dataset.from_generator(rand_gen, args=[train_files],
                                              output_types=tf.string)
    
    # data load and augment
    train_ds = train_ds.map(lambda x: tf.py_function(func=process_mat_train, 
                                                     inp=[x, True] + arg_list, 
                                                     Tout=(tf.float32, tf.float32)),
                            num_parallel_calls=tf.data.AUTOTUNE
                            )

    train_ds = train_ds.batch(batch_size=batch_sz)
    print('Train dataset batches =', train_num//batch_sz) 
    


    # Validation set    
    val_num = len(val_files)
    print('Val dataset n =', val_num) 
    
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    
    val_ds = val_ds.map(lambda x: tf.py_function(func=process_mat_train, 
                                                 inp=[x, False] + arg_list, 
                                                 Tout=(tf.float32, tf.float32)),
                        num_parallel_calls=tf.data.AUTOTUNE
                        )
    
    val_ds = val_ds.batch(batch_size=batch_sz)

    print('Val dataset batches =', val_num//batch_sz)
    
    print('Model learning rate:', model.optimizer.learning_rate)

    
    if class_weights is not None:
        
        print('Using class weights of', class_weights)
        
        weights = class_weights/tf.reduce_sum(class_weights)
        print('Normed:', weights)

        train_ds = train_ds.map(lambda x, y: add_sample_weights(x,y,weights))
        
        # Need to add sample weights to val?
        val_ds = val_ds.map(lambda x, y: add_sample_weights(x,y,weights))
    

    # Do a check before proceeding
    viewTrainVal(train_ds, val_ds)
    
    viewTest(model, val_files, arg_list, cur_i=0)
        
    if ask:
        val = input("Proceed? (y): ")
        if val != 'y':
            print('Quitting...')
            return
        
    # Make output folder for model        
    os.makedirs(out_folder, exist_ok=False) #don't overwrite!
        

    # Define callbacks -------------------------------------------------------

    # Early stopping
    earlystopper = EarlyStopping(patience=patience, verbose=1, monitor=monitoring, mode=monitor_mode)
    
    # Checkpointer to save model
    cur_date = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    checkpointer = ModelCheckpoint(out_folder+'/'+cur_date+'-TF'+tf.__version__\
                                    +'-Net-CP{epoch:03d}-{loss:.3E}-{val_loss:.3E}'+addiFmt+'.h5',
                                    verbose=1, 
                                    save_best_only=True,
                                    monitor=monitoring, mode=monitor_mode)
        
    # Tensorboard logs
    tboard_logs = out_folder+'/logs'
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tboard_logs,
                                                 histogram_freq = 1,
                                                 profile_batch = (2,5) #profiles batch 2-5
                                                 )
    
    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            
            viewTest( self.model, train_files, arg_list, cur_i=None, cur_z=None, miscTxt='T_Epoch '+str(epoch+1) )
            viewTest( self.model, val_files, arg_list, cur_i=None, cur_z=None, miscTxt='V_Epoch '+str(epoch+1) )

            print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
    
    
    
    #--------------------------------------------------------------------------
    # Train!
    results = model.fit(
        train_ds, 
        validation_data = val_ds, 
        epochs = epoch_n, 
        callbacks = [earlystopper, 
                     checkpointer, 
                     tboard_callback, 
                     DisplayCallback()],
        workers=12, use_multiprocessing=True)

    #--------------------------------------------------------------------------
    end_time = time.time() 
    end_time_str = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime(end_time))
    
    print(start_time_str)
    print(end_time_str)
    elapsed_secs = end_time-start_time
    print('Elapsed secs:',elapsed_secs)

    # create timing dict
    timing = {'start': start_time_str,
              'end': end_time_str, 
              'elapsed_secs': elapsed_secs}          
            
    # open file for writing
    f = open(out_folder+'/'+cur_date+"_results.txt","w")
    f.write(str([results.history, timing]))
    f.close()
    
    # display loss graph
    viewLoss(results.history, timing)    
    
    # display example val data using best model
    models = glob.glob(out_folder+'/*.h5')
    
    # save final model regardless of monitoring
    tf.keras.models.save_model(model, out_folder+'/'+cur_date+'-TF'+tf.__version__\
                                    +'-Net-CP'+'{:03d}'.format(model.history.epoch[-1])+'.h5')
    
    # pull the "best" model from before the final model
    model = load_model(models[-1], custom_objects={'CustomMetric_Dice': tf.keras.metrics.Accuracy})
    
    viewTest(model, val_files, arg_list, cur_i=0)
    viewTest(model, val_files, arg_list, cur_i=len(val_files)-1)
    viewTest(model, val_files, arg_list, cur_i=round(len(val_files)/2))

    # Save "best" model as onnx file
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    onnx.save(onnx_model, models[-1][:-3] + '.onnx')
    print('Onnx file saved!')

    print('Training complete!')

    return




# Main
if __name__ == '__main__':
    
    # ----------------------------------------------------------------------------
    # Set up data
    # ----------------------------------------------------------------------------
    start_time = time.time()
    start_time_str = time.strftime("%Y%m%d_%H%M-", time.localtime(start_time))
    print('-------------------------------------')
    print('Running main script at', start_time_str, '\n')
    
    # DATA PATH
    data_paths = [
        r'D:\Serge 231\Rib\Data\Rib_Patches'
                 ]
    
    name_prefix = '20230410_512_2_RibSeg_'
    
    all_grps = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
    
    models_list = [1]
    
    # Loop over models
    for model_n in range(len(models_list)):
        train_grps = all_grps.copy()
        train_grps.remove(train_grps[models_list[model_n]-1])
        val_grps = [all_grps[models_list[model_n]-1]]
    
        name_str = name_prefix + val_grps[0]
    
        # GET TRAINING FILES
        train_files = []
        for data_path in data_paths:
            for train_grp in train_grps:
                cur_train_files =   glob.glob(data_path + '/*/' + train_grp + '/*/*.mat')
                train_files = train_files + cur_train_files
                print(data_path)
                print(len(cur_train_files))
                        
        print('Train files:', len(train_files))
        
        # GET VAL FILES
        val_files = []
        for data_path in data_paths:
            for val_grp in val_grps:
                cur_val_files =   glob.glob(data_path + '/*/' + val_grp + '/*/*.mat')
                val_files = val_files + cur_val_files
                print(data_path)
                print(len(cur_val_files))
                            
        print('Val files:', len(val_files))
    
    
        # Skipping?
        print('\nWith skips:')
        skip = 3
        train_files = train_files[::skip]
        val_files = val_files[::skip]
    
        print('\nTotal Train files:', len(train_files))
        print('Total Val files:', len(val_files))
        
        # Random shuffle? Set seed so it is consistent
        random.seed(99)
        random.shuffle(train_files)
        random.shuffle(val_files)
        print('Train filenames:',train_files[0:3])
        print('Val filenames:',val_files[0:3])
    
        # Build Unet
        sz = [512, 512, 1]
        depth = 4
        filters = 16
        num_classes = 2
        drop_rate = 0.5
        batch_norm_bool = True
        final_bias = 0.5
        final_activation = 'softmax'
        
        # Generate new model
        model = unet2D(sz, depth, filters, num_classes, \
                       drop_rate, batch_norm_bool, final_bias, \
                           final_activation)
    
    
        net = '2'
    
        # Training info
        epoch_n = 40
        batch_sz = 16
        init_learning_rate = 3e-4
    
        out_folder = r'D:\Serge 231\Rib\Models\Rib_Models'
        out_folder_model = start_time_str+'-'+net+'-d'+str(depth)+'f'+str(filters)+\
                's'+str(sz[0])+'b'+str(batch_sz) + '-' + name_str
        out_folder_full = out_folder+'/'+out_folder_model
        
        # Learning rate scheduler
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            init_learning_rate,
            decay_steps = round(len(train_files)/batch_sz)*1,
            decay_rate = 0.9,
            staircase = True
            )
    
        # Compile the model
        model.compile(    optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule), \
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=[tf.keras.metrics.CategoricalAccuracy(), \
                                   CustomMetric_Dice(1)] )
        addiFmt = '-{custom_dice1:.3f}-{val_custom_dice1:.3f}-'
        class_weights = None
        
        # Lung: [-1150, 350], level_rng=[-500, -300], window_rng=[1200, 1600],
        # Pe: [-250, 450], window level: [700, 100], level_rng=[0, 200], window_rng=[600, 800],
    
        arg_list = build_arg_list(num_classes=num_classes,
                                  window_rng_default=[0, 4095],
                                  window_rng_type='norm-clip',
                                  level_rng=[2050, 2150], window_rng=[4095, 4195],
                                  col_flip_prob=0.5, ang_rng=[-10, 10],
                                  blur_prob=0.05, blur_sigma_rng=[0.5,0.7],
                                  sharp_prob=0.05, sharp_sigma_rng=[0.5,0.9],
                                  noise_prob=0.025, noise_perc=0.01,
                                  one_hot_bool=True,
                                  )
        
        print(arg_list)
        
        # Train model!
        train(model, train_files, val_files, out_folder_full, epoch_n, batch_sz,
              addiFmt,
              monitoring='val_loss', monitor_mode='min', patience=15,
              class_weights=class_weights,
              arg_list=arg_list)
        
        print('Done with training model '+str(models_list[model_n]))
        




