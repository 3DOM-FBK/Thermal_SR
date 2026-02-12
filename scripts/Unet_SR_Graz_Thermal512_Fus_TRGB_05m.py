# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:37:09 2021

@author: salim
"""


# from model1 import *
from datetime import datetime

import glob
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import vstack, hstack
from numpy.random import randn, rand
from numpy.random import randint, permutation
import random
import os

from tensorflow.keras import applications
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
# from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np 
import os
import math
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow.keras.backend as K
from skimage.metrics import structural_similarity as ssim
import cv2
# from PIL import Image
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# dim=64  # 224
# dimG=128
# bands = 1

def psnr(img1, img2,PIXEL_MAX):
    mse = np.mean( (img1.astype("float") - img2.astype("float")) ** 2 )
    # print(mse)
    if mse == 0:
        return 100
    # PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def mse(imageA, imageB, bands):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def rmse(imageA, imageB, bands):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
	err = np.sqrt(err)
	return err
	
	# return the MSE, the lower the error, the more "similar"
	# the two im
# div=4
bands = 4
div_RGB=12.75
# max_val=80

def custom_loss(y_true, y_pred):
   SSIML=tf.image.ssim(y_true,y_pred,max_val=1)
#    loss1 = 2*(1-SSIML)
   loss1 = (1-SSIML)
   loss22 = tf.keras.metrics.MAE(y_true,y_pred)
   loss23 = tf.reduce_mean(loss22, axis=-1)
   loss2 = tf.reduce_mean(loss23, axis=-1)
   return (loss1 + loss2)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.he_normal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    # return tf.nn.relu(x)
    return layers.ReLU()(x)

# def unet1(input_size,input_size1):
def unet1(input_size):
    inputs = Input(input_size)  # 0
    # input2 = Input(input_size1)
    # layers 1-5
    conv1_1 = convolution_block(inputs, num_filters=16, kernel_size=3, dilation_rate=1)  # 1
    conv1_2 = convolution_block(inputs, num_filters=16, kernel_size=3, dilation_rate=12)  # 1
    conv1_3 = convolution_block(inputs, num_filters=16, kernel_size=3, dilation_rate=24)  # 1
    conv1_4 = convolution_block(inputs, num_filters=16, kernel_size=3, dilation_rate=36)  # 1
    merge1 = concatenate([conv1_1,conv1_2,conv1_3,conv1_4], axis = 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(merge1) # 5
    # layers 6-10
    conv2_1 = convolution_block(pool1, num_filters=32, kernel_size=3, dilation_rate=1)  # 1
    conv2_2 = convolution_block(pool1, num_filters=32, kernel_size=3, dilation_rate=6)  # 1
    conv2_3 = convolution_block(pool1, num_filters=32, kernel_size=3, dilation_rate=12)  # 1
    conv2_4 = convolution_block(pool1, num_filters=32, kernel_size=3, dilation_rate=18)  # 1
    merge2 = concatenate([conv2_1,conv2_2,conv2_3,conv2_4], axis = 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(merge2) # 5
    # layers 11-16
    conv3_1 = convolution_block(pool2, num_filters=64, kernel_size=3, dilation_rate=1)  # 1
    conv3_2 = convolution_block(pool2, num_filters=64, kernel_size=3, dilation_rate=3)  # 1
    conv3_3 = convolution_block(pool2, num_filters=64, kernel_size=3, dilation_rate=6)  # 1
    conv3_4 = convolution_block(pool2, num_filters=64, kernel_size=3, dilation_rate=9)  # 1
    merge3 = concatenate([conv3_1,conv3_2,conv3_3,conv3_4], axis = 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(merge3) # 5
    # layers 17-22
    conv4_1 = convolution_block(pool3, num_filters=128, kernel_size=3, dilation_rate=1)  # 1
    conv4_2 = convolution_block(pool3, num_filters=128, kernel_size=3, dilation_rate=2)  # 1
    conv4_3 = convolution_block(pool3, num_filters=128, kernel_size=3, dilation_rate=4)  # 1
    conv4_4 = convolution_block(pool3, num_filters=128, kernel_size=3, dilation_rate=6)  # 1
    merge4 = concatenate([conv4_1,conv4_2,conv4_3,conv4_4], axis = 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(merge4) # 5
    # # layers 23-28
    conv5_1 = convolution_block(pool4, num_filters=128, kernel_size=3, dilation_rate=1)  # 1
    conv5_2 = convolution_block(pool4, num_filters=128, kernel_size=3, dilation_rate=2)  # 1
    conv5_3 = convolution_block(pool4, num_filters=128, kernel_size=3, dilation_rate=3)  # 1
    conv5_4 = convolution_block(pool4, num_filters=128, kernel_size=3, dilation_rate=4)  # 1
    merge5 = concatenate([conv5_1,conv5_2,conv5_3,conv5_4], axis = 3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(merge5) # 5
    # # layers 23-28
    conv5 = convolution_block(pool5, num_filters=1024, kernel_size=3, dilation_rate=1)  # 1
    conv5 = convolution_block(conv5, num_filters=1024, kernel_size=3, dilation_rate=1)  # 1
    # conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) # 25
    # Norm5 = BatchNormalization()(conv5) # 26
    # Norm5 = Dropout(0.05)(conv5) # 26
    # drop5 = Dropout(0.25)(Norm5) # 27
    # pool5 = MaxPooling2D(pool_size=(2, 2))(drop5) # 28
    # conv55 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5) # 25
    # conv55 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv55) # 25

    # upsampling part
    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv55))
    # merge6 = concatenate([Norm5,up6], axis = 3)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    # Norm6 = BatchNormalization()(conv6)
    # drop6 = Dropout(0.25)(Norm6)
    
    up7=UpSampling2D(size = (2,2))(conv5)
    conv7 = convolution_block(up7, num_filters=512, kernel_size=3, dilation_rate=1)  # 1
    merge7 = concatenate([merge5,conv7], axis = 3)
    conv7_1 = convolution_block(merge7, num_filters=128, kernel_size=3, dilation_rate=1)  # 1
    conv7_2 = convolution_block(merge7, num_filters=128, kernel_size=3, dilation_rate=2)  # 1
    conv7_3 = convolution_block(merge7, num_filters=128, kernel_size=3, dilation_rate=3)  # 1
    conv7_4 = convolution_block(merge7, num_filters=128, kernel_size=3, dilation_rate=4)  # 1
    merge77 = concatenate([conv7_1,conv7_2,conv7_3,conv7_4], axis = 3)
    
    up8=UpSampling2D(size = (2,2))(merge77)
    conv8 = convolution_block(up8, num_filters=512, kernel_size=3, dilation_rate=1)  # 1
    merge8 = concatenate([merge4,conv8], axis = 3)
    conv8_1 = convolution_block(merge8, num_filters=128, kernel_size=3, dilation_rate=1)  # 1
    conv8_2 = convolution_block(merge8, num_filters=128, kernel_size=3, dilation_rate=2)  # 1
    conv8_3 = convolution_block(merge8, num_filters=128, kernel_size=3, dilation_rate=4)  # 1
    conv8_4 = convolution_block(merge8, num_filters=128, kernel_size=3, dilation_rate=6)  # 1
    merge88 = concatenate([conv8_1,conv8_2,conv8_3,conv8_4], axis = 3)

    up9=UpSampling2D(size = (2,2))(merge88)
    conv9 = convolution_block(up9, num_filters=256, kernel_size=3, dilation_rate=1)  # 1
    merge9 = concatenate([merge3,conv9], axis = 3)
    conv9_1 = convolution_block(merge9, num_filters=64, kernel_size=3, dilation_rate=1)  # 1
    conv9_2 = convolution_block(merge9, num_filters=64, kernel_size=3, dilation_rate=3)  # 1
    conv9_3 = convolution_block(merge9, num_filters=64, kernel_size=3, dilation_rate=6)  # 1
    conv9_4 = convolution_block(merge9, num_filters=64, kernel_size=3, dilation_rate=9)  # 1
    merge99 = concatenate([conv9_1,conv9_2,conv9_3,conv9_4], axis = 3)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    up10=UpSampling2D(size = (2,2))(merge99)
    conv10 = convolution_block(up10, num_filters=128, kernel_size=3, dilation_rate=1)  # 1
    merge10 = concatenate([merge2,conv10], axis = 3)
    conv10_1 = convolution_block(merge10, num_filters=32, kernel_size=3, dilation_rate=1)  # 1
    conv10_2 = convolution_block(merge10, num_filters=32, kernel_size=3, dilation_rate=6)  # 1
    conv10_3 = convolution_block(merge10, num_filters=32, kernel_size=3, dilation_rate=12)  # 1
    conv10_4 = convolution_block(merge10, num_filters=32, kernel_size=3, dilation_rate=18)  # 1
    merge10_2 = concatenate([conv10_1,conv10_2,conv10_3,conv10_4], axis = 3)
    
    up11=UpSampling2D(size = (2,2))(merge10_2)
    conv11 = convolution_block(up11, num_filters=64, kernel_size=3, dilation_rate=1)  # 1
    merge11 = concatenate([merge1,conv11], axis = 3)
    conv11_1 = convolution_block(merge11, num_filters=16, kernel_size=3, dilation_rate=1)  # 1
    conv11_2 = convolution_block(merge11, num_filters=16, kernel_size=3, dilation_rate=12)  # 1
    conv11_3 = convolution_block(merge11, num_filters=16, kernel_size=3, dilation_rate=18)  # 1
    conv11_4 = convolution_block(merge11, num_filters=16, kernel_size=3, dilation_rate=36)  # 1
    merge11_2 = concatenate([conv11_1,conv11_2,conv11_3,conv11_4], axis = 3)



    conv11 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11_2)
    conv11 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    # Norm11 = BatchNormalization()(conv11)
    # drop11 = Dropout(0.25)(conv11)

    
    conv15 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    model = Model(inputs, conv15)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    return model

cwd = os.getcwd()
cwd2='/scratch/3DOM/smalek/SR_Thermal/'

dim=512  # 224
dimG=512
# # N=1024   # number total of training images 
# # N0=150  # number of samples Tr of class0
# # N1=874  # number of samples Tr of class1
# # IMG_DIM = (224, 224)
# IMG_DIM = (dim, dim)
input_shape = (dim, dim, bands)
input_shape1 = (dim, dim, 3)
# input_shape2 = (dimG, dimG, 1)
# model1 = unet1(input_shape,input_shape)
model1 = unet1(input_shape)

model1.compile(optimizer = Adam(learning_rate = 5e-4), loss = custom_loss, metrics = ['RootMeanSquaredError'])
# model1.compile(optimizer = Adam(lr = 1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics = ['RootMeanSquaredError'])
model1.summary() 

print('-------------------------------------------------curent path=',cwd)

print('----- read data1 -------')
# train_dir = askdirectory(title = 'The directory containing train files of class 1 (with defect)!') 
dir_trRGB=cwd +'/Graz_dataset_512/Train/Input_RGB_05m/'
# dir_trRGB=cwd +'/Graz_dataset_512/Train/Input_grd_10m/'
dir_trT=cwd +'/Graz_dataset_512/Train/Input_T_30m_up05/'
dir_trGT=cwd +'/Graz_dataset_512/Train/Output_T_05m/'
train_files = glob.glob(os.path.join(dir_trT, '*.npy'))
train_files_name = [fn.split('/')[-1].split('.npy')[0].strip() for fn in train_files]

dir_valRGB=cwd +'/Graz_dataset_512/Val/Input_RGB_05m/'
dir_valT=cwd +'/Graz_dataset_512/Val/Input_T_30m_up05/'
dir_valGT=cwd +'/Graz_dataset_512/Val/Output_T_05m/'
val_files = glob.glob(os.path.join(dir_valT, '*.npy'))
val_files_name = [fn.split('/')[-1].split('.npy')[0].strip() for fn in val_files]

# dir_tsRGB=cwd +'/Graz_dataset_512/Test/Input_gray_05m/'
# dir_tsT=cwd +'/Graz_dataset_512/Test/Input_T_30m_org/'
# dir_tsGT=cwd +'/Graz_dataset_512/Test/Output_T_05m_org/'
# test_files = glob.glob(os.path.join(dir_tsT, '*.npy'))
# test_files_name = [fn.split('/')[-1].split('.npy')[0].strip() for fn in test_files]
print('------------------------lenth tr = ',str(len(train_files)))
print('------------------------lenth val = ',str(len(val_files)))
# print('------------------------lenth ts = ',str(len(test_files)))

N = len(train_files_name)
Nv = len(val_files_name)
# Ns = len(test_files_name)

epochs1 = 1
batch_size1 = 4
train_imgs=zeros((batch_size1,dim,dim,bands))
train_labels=zeros((batch_size1,dim,dim,1))
val_imgs=zeros((1,dim,dim,bands))
# val_labels=zeros((1,dim,dim,1))
# test_imgs=zeros((1,dim,dim,bands))
# test_labels=zeros((1,dim,dim,1))
# num_classes = 2
# epochs1 = 1
# N=1200
# N1=100 #800
N2=batch_size1
N1=int(N/N2)

iter0=0
# iter0=12

epochs_max=300

MAE_min=9999999999.99
# MAE_min=0.120795549300

stop=0
i=-1
max_nb_min=13
nb_min=0
val_Acc=np.zeros((300,4))
# ts_Acc=np.zeros((300,4))
tr_Acc=np.zeros((300,2))
time_Tr=np.zeros((300,2))
print('start init tr1 --------------------------------------------------------------')

gpu_available = tf.test.is_gpu_available()
print('gpu_available=',gpu_available)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

while((i<epochs_max - iter0) and (stop==0)):
# for i1 in range(1):
    i=i+1
    start1=datetime.now()
    iter1=epochs1*(i+1)+iter0
    # iter1=epochs1*(i+1)
    start=datetime.now()
    rand_p=permutation(N)
    idi=0
    lossTr=0
    for i21 in range(N1):
    # for i21 in range(50):
        for i22 in range(N2):
        
            i2=rand_p[idi]
            idi=idi+1
            # print(dir_tr+files_name[i2][:]+'.npy')
            # print(files_name[i2][:-4])
            train_imgs[i22,:,:,0]=np.load(dir_trT+train_files_name[i2][:]+'.npy')
            # print(' train_imgs[i22,:,:,0] min max =', train_imgs[i22,:,:,0].min(),' ', train_imgs[i22,:,:,0].max())
            img=cv2.imread(dir_trRGB+train_files_name[i2][:]+'.png')
            # image0 = Image.open(dir_trRGB+train_files_name[i2][:]+'.png')
            # b, g, r = image0.split()
            # image = Image.merge("RGB", (r, g, b))
            # img=np.array(image)
            
            # img=img/255
            # img=img/12.75
            img=img/div_RGB
            # print('img min max =',img.min(),' ',img.max())
            train_imgs[i22,:,:,1:4] =img
            train_labels[i22,:,:,0]=np.load(dir_trGT+train_files_name[i2][:]+'.npy')
            # print('train_labels[i22,:,:,0] min max =',train_labels[i22,:,:,0].min(),' ',train_labels[i22,:,:,0].max())
        
        # train_labels=helpers.reverse_one_hot(helpers.one_hot_it(imagt, label_values))
        

        

        # train_labels=np.reshape(imagt,(1,dimG,dimG,1))
        # train_imgs=np.reshape(train_imgs,(1,dim,dim,1))
        
        print('Unet_SR_Graz_Thermal512_Fus_TRGB_05m iteration number   ',iter1,' sub-iter = ',i21+1)
        history_model1 = model1.fit(train_imgs, train_labels,
                    epochs=epochs1, # shuffle=false,
                    batch_size=batch_size1) 
        a=history_model1.history['loss']
        lossTr=lossTr+a[0]
    tr_Acc[iter1-1,0]=iter1-1
    tr_Acc[iter1-1,1]=lossTr/N1
    stopTr=datetime.now()
    timeTr=stopTr-start1
    time_Tr[iter1-1,0]=iter1-1
    time_Tr[iter1-1,1]=timeTr.seconds    
    np.save(cwd +'/Tr_Acc_Unet_SR_Graz_Thermal512_Fus_TRGB_05m',tr_Acc)
    np.save(cwd +'/Tr_runtime_Unet_SR_Graz_Thermal512_Fus_TRGB_05m',time_Tr)        

    if(iter1==1):
        model1.compile(optimizer = Adam(learning_rate = 2e-4), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==2):
        model1.compile(optimizer = Adam(learning_rate = 1e-4), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==3):
        model1.compile(optimizer = Adam(learning_rate = 5e-5), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==5):
        model1.compile(optimizer = Adam(learning_rate = 2e-5), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==8):
        model1.compile(optimizer = Adam(learning_rate = 1e-5), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==12):
        model1.compile(optimizer = Adam(learning_rate = 5e-6), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==18):
        model1.compile(optimizer = Adam(learning_rate = 2e-6), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==25):
        model1.compile(optimizer = Adam(learning_rate = 1e-6), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==35):
        model1.compile(optimizer = Adam(learning_rate = 5e-7), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==50):
        model1.compile(optimizer = Adam(learning_rate = 2e-7), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==80):
        model1.compile(optimizer = Adam(learning_rate = 1e-7), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==120):
        model1.compile(optimizer = Adam(learning_rate = 5e-8), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==200):
        model1.compile(optimizer = Adam(learning_rate = 2e-8), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==300):
        model1.compile(optimizer = Adam(learning_rate = 1e-8), loss = custom_loss, metrics = ['RootMeanSquaredError'])
 
    
    # (train_imgs, train_labels)=shuffle(train_imgs, train_labels);
    print('evaluate validation**********')
    RMSEv=[]
    PSNRv=[]
    SSIMv=[]
    for idi in range(Nv):
        val_imgs[0,:,:,0]=np.load(dir_valT+val_files_name[idi][:]+'.npy')
        img=cv2.imread(dir_valRGB+val_files_name[idi][:]+'.png')
        # image0 = Image.open(dir_valRGB+val_files_name[idi][:]+'.png')
        # b, g, r = image0.split()
        # image = Image.merge("RGB", (r, g, b))
        # img=np.array(image)        
        # # img=img/255
        # # img=img/12.75
        img=img/div_RGB
        val_imgs[0,:,:,1:4] =img
        val_labels=np.load(dir_valGT+val_files_name[idi][:]+'.npy')
        predicted = model1.predict(val_imgs,verbose=0)
        predicted = np.reshape(predicted,(dim,dim))
        RMSE = rmse(predicted,val_labels,bands) # mae(imagt255,predicted255,bands)
        # PSNR = tf.image.psnr(, predicted255 , max_val=255)
        PSNR=psnr(predicted,val_labels,1)
        SSIM = ssim(val_labels, predicted, data_range=predicted.max() - predicted.min(), multichannel=False)
        RMSEv.append(RMSE)
        PSNRv.append(PSNR)
        SSIMv.append(SSIM)

    val_Acc[iter1-1,0]=iter1-1
    val_Acc[iter1-1,1] = np.mean(RMSEv)
    val_Acc[iter1-1,2] = np.mean(PSNRv)
    val_Acc[iter1-1,3] = 100*(np.mean(SSIMv)) 
    np.save(cwd +'/Val_Acc_Unet_SR_Graz_Thermal512_Fus_TRGB_05m',val_Acc)

    acc1=(val_Acc[iter1-1,1] + 2*(1-(np.mean(SSIMv))))/3
    if(acc1<MAE_min):
        MAE_min=acc1
        nb_min = 0
        model1.save(cwd +'/Unet_SR_Graz_Thermal512_Fus_TRGB_05m_'+str(iter1)+'it.keras')
    else:
        nb_min=nb_min+1
    if(nb_min>max_nb_min):
        stop=1             

    # print('evaluate test**********')
    # RMSEs=[]
    # PSNRs=[]
    # SSIMs=[]
    # for idi in range(Ns):
    #     test_imgs[0,:,:,0]=np.load(dir_tsT+test_files_name[idi][:]+'.npy')
    #     img=cv2.imread(dir_tsRGB+test_files_name[idi][:]+'.png')
    #     # img=img/255
    #     img=img/12.75
    #     test_imgs[0,:,:,1] =img[:,:,0]         
    #     test_labels=np.load(dir_tsGT+test_files_name[idi][:]+'.npy')
    #     predicted = model1.predict(test_imgs,verbose=0)
    #     predicted = np.reshape(predicted,(dim,dim))
    #     RMSE = rmse(predicted,test_labels,bands) # mae(imagt255,predicted255,bands)
    #     # PSNR = tf.image.psnr(, predicted255 , max_val=255)
    #     PSNR=psnr(predicted,test_labels,1)
    #     SSIM = ssim(predicted,test_labels, multichannel=False)
    #     RMSEs.append(RMSE)
    #     PSNRs.append(PSNR)
    #     SSIMs.append(SSIM)

    # ts_Acc[iter1-1,0]=iter1-1
    # ts_Acc[iter1-1,1] = np.mean(RMSEs)
    # ts_Acc[iter1-1,2] = np.mean(PSNRs)
    # ts_Acc[iter1-1,3] = 100*(np.mean(SSIMs))  
    # np.save(cwd +'/Ts_Acc_Unet_SR_Graz_Thermal512_Fus_TRGB_05m',ts_Acc)

    
    # run_time1=datetime.now()-start
    print('time it = ',datetime.now()-start)


