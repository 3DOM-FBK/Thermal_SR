# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:37:09 2021

@author: salim
"""

# example UNET on Pet dataset

# from model1 import *
# reset
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import glob
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import vstack, hstack
from numpy.random import randn, rand
from numpy.random import randint, permutation
import random
import os
# import matplotlib.pyplot as plt
import math
# from scipy import interpolate
import cv2
import copy
import rasterio

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras.utils import plot_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
div_im=12.75
max_val=80

### ************ to select which model ************ 
# selected_model=1 #Graz
# selected_model=2 #Ferrara
selected_model=3 #Graz_Ferrara
### ************************************************* 

def custom_loss(y_true, y_pred):
   SSIML=tf.image.ssim(y_true,y_pred,max_val=150)
   # loss = math_ops.mean(diff, axis=1) #mean over last dimension
   loss1 = 2*(1-SSIML)
   loss22 = tf.keras.metrics.MAE(y_true,y_pred)
   loss23 = tf.reduce_mean(loss22, axis=-1)
   loss2 = tf.reduce_mean(loss23, axis=-1)
   return (loss1 + loss2)

def psnr(img1, img2, PIXEL_MAX):
    mse = np.mean( (img1.astype("float") - img2.astype("float")) ** 2 )
    # print(mse)
    if mse == 0:
        return 100
    # PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def mse(imageA, imageB, bands1):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands1)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def rmse(imageA, imageB, bands1):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands1)
	err = np.sqrt(err)
	return err
	

bands = 5
cwd = os.getcwd()
# cwd2='/scratch/3DOM/smalek/SR_Thermal'
# cwd3='/scratch/3DOM/smalek/SR_Thermal/Ferrara'

dim=512  



if(selected_model==1):
    model1=load_model(cwd +'/Unet_SR_Graz_Thermal512_Fus_TRGBEmissiv_05m_9it.keras', custom_objects={'custom_loss': custom_loss}) # Graz mdl
else:
    if(selected_model==2):
        model1=load_model(cwd +'/Unet_SR_Ferrara_Thermal512_Fus_TRGBEmissiv_05m_9it.keras', custom_objects={'custom_loss': custom_loss}) # Ferrara mdl
    else:
        if(selected_model==3):
            model1=load_model(cwd +'/Unet_SR_Graz_Ferrara_Thermal512_Fus_TRGBEmissiv_05m_23it.keras', custom_objects={'custom_loss': custom_loss}) # Graz_Ferrara mdl




im_T00=rasterio.open(cwd+'/Aligned_Graz_data_Ts_part/Ts_Part_Linear_up05_Graz_Ts_LC08_190027_20210912_LST_C_small_aoi_2.tif')
im_GT00=rasterio.open(cwd+'/Aligned_Graz_data_Ts_part/Ts_Part_Graz_day_surface_temperature_UTM33_mask_small_50cm_celcius_2.tif')
im_RGB05m0=rasterio.open(cwd+'/Aligned_Graz_data_Ts_part/Ts_Part_RGB_mosaic_50cm_2.tif')
im_Emmiss05m0=rasterio.open(cwd+'/Aligned_Graz_data_Ts_part/Ts_Part_graz_emissivity_50cm_masked_32633_2.tif')

profileT=im_T00.profile
im_T=im_T00.read(1)
sz1T=im_T.shape[0]
sz2T=im_T.shape[1]

im_GT=im_GT00.read(1)
im_B05m=im_RGB05m0.read(1)
im_G05m=im_RGB05m0.read(2)
im_R05m=im_RGB05m0.read(3)
im_Emmiss05m=im_Emmiss05m0.read(1)
im_Emmiss05m=(im_Emmiss05m-0.7)*65

sz01=im_GT.shape[0] # height
sz02=im_GT.shape[1] # width
# im_T=im_T00 #cv2.resize(im_T0,(sz02,sz01),cv2.INTER_LINEAR) already prepared
print('im_GT shape=',im_GT.shape)

im_RGB05m=np.uint8(np.zeros((sz01,sz02,3)))
im_RGB05m[:,:,0]=im_R05m
im_RGB05m[:,:,1]=im_G05m
im_RGB05m[:,:,2]=im_B05m
im_RGB05m=im_RGB05m/div_im

# im_RGB05m2=np.uint8(np.zeros((3,sz01,sz02)))
# im_RGB05m2[2,:,:]=im_R05m
# im_RGB05m2[1,:,:]=im_G05m
# im_RGB05m2[0,:,:]=im_B05m



# dimTs=int(sz01*0.2)
# ima1org=im_T[sz01-dimTs:sz01,:]
# ima1=im_T[sz01-dimTs:sz01,:]
# ima2=im_RGB05m[sz01-dimTs:sz01,:] 
# ima22=im_RGB05m2[:,sz01-dimTs:sz01] 
# # ima3=im_Emmiss05m[sz01-dimTs:sz01,:]  # just emmissiv or emmissiv+land cover

# # ima1=im_Tn[sz01-dimTs:sz01,:]
# # ima2=im_RGB[sz01-dimTs:sz01,:] 
# im_GT=im_GT0[sz01-dimTs:sz01,:]
# print('ima shape=',ima1.shape)
sz1=sz01 #im_GT.shape[0]; # height
sz2=sz02 #im_GT.shape[1]; # width


# profileT=im_T00.profile
# print(profileT)
# profileRGB=im_RGB05m0.profile
# affine=profileT["transform"]
# px=affine[2]
# resolution=0.5
# py=affine[5]-(sz01-int(sz01*0.2))*resolution
# affine=[0.5,affine[1],affine[2],affine[3],-0.5,py]
# profileT.update(
#     width= sz2, 
#     height= sz1, 
#     transform= affine)

# profileRGB.update(
#     width= sz2, 
#     height= sz1, 
#     transform= affine)

# with rasterio.open(cwd+'/prediction/Linear_Graz_Ts_LC08_190027_20210912_LST_C_small_aoi_2.tif', 'w', **profileT) as dst:
#     dst.write(ima1org,1)
# with rasterio.open(cwd+'/prediction/RGB.tif', 'w', **profileRGB) as dst:
#     dst.write(ima22,[1,2,3])
# with rasterio.open(cwd+'/prediction/Graz_day_Ts_surface_temperature_UTM33_mask_small_50cm_celcius_2.tif', 'w', **profileT) as dst:
#     dst.write(im_GT,1)        


ima=np.zeros((sz1,sz2,bands))
ima[:,:,0]=im_T
# ima[:,:,1:bands]=ima2
ima[:,:,1:4]=im_RGB05m
ima[:,:,4]=im_Emmiss05m
nb1=np.uint16(sz1/dim)
nb2=np.uint16(sz2/dim)

resid1=np.uint16(sz1%dim)
resid2=np.uint16(sz2%dim)
print('nb1 nb2',nb1,' ',nb2)
if(resid1 > (dim/2)):
    nb21=nb1+1
else:
    nb21=nb1
if(resid2 > (dim/2)):
    nb22=nb2+1
else:
    nb22=nb2

pred_label=np.zeros((sz1,sz2))

for nth in range(1):
    for i1 in range(nb1):
        print('1): ',i1,' / ',nb1)
        for i2 in range(nb2):
            ima_crop=ima[i1*dim:(i1+1)*dim,i2*dim:(i2+1)*dim,:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[i1*dim:(i1+1)*dim,i2*dim:(i2+1)*dim]=predicted
            
    if(resid1>0):
        i1=nb1
        lim1=int(((nb1*dim)+(sz1-dim))/2)
        for i2 in range(nb2):
            ima_crop=ima[sz1-dim:sz1,i2*dim:(i2+1)*dim,:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[lim1:sz1,i2*dim:(i2+1)*dim]=predicted[dim-(sz1-lim1):dim,:]

        for i2 in range(nb22-1):
            ima_crop=ima[sz1-dim:sz1,i2*dim+int(dim/2):(i2+1)*dim+int(dim/2),:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[lim1:sz1,i2*dim+int(3*dim/4):(i2+1)*dim+int(dim/4)]=predicted[dim-(sz1-lim1):dim,int(dim/4):int(3*dim/4)]

            
    if(resid2>0):
        i2=nb2
        lim2=int(((nb2*dim)+(sz2-dim))/2)
        for i1 in range(nb1):
            ima_crop=ima[i1*dim:(i1+1)*dim,sz2-dim:sz2,:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[i1*dim:(i1+1)*dim,lim2:sz2]=predicted[:,dim-(sz2-lim2):dim]

        for i1 in range(nb21-1):
            ima_crop=ima[i1*dim+int(dim/2):(i1+1)*dim+int(dim/2),sz2-dim:sz2,:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[i1*dim+int(3*dim/4):(i1+1)*dim+int(dim/4),lim2:sz2]=predicted[int(dim/4):int(3*dim/4),dim-(sz2-lim2):dim]

        
    if( (resid1>0) & (resid2>0) ):
        i1=nb1
        i2=nb2
        lim1=int(((nb1*dim)+(sz1-dim))/2)
        lim2=int(((nb2*dim)+(sz2-dim))/2)

        ima_crop=ima[sz1-dim:sz1,sz2-dim:sz2,:]    
        ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dim,dim))
        pred_label[lim1:sz1,lim2:sz2]=predicted[dim-(sz1-lim1):dim,dim-(sz2-lim2):dim]

    for i1 in range(nb21-1):
        print('2): ',i1,' / ',nb21-1)
        for i2 in range(nb2):
            ima_crop=ima[i1*dim+int(dim/2):(i1+1)*dim+int(dim/2),i2*dim:(i2+1)*dim,:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[i1*dim+int(3*dim/4):(i1+1)*dim+int(dim/4),i2*dim:(i2+1)*dim]=predicted[int(dim/4):int(3*dim/4),:]
    
    for i1 in range(nb1):
        print('3): ',i1,' / ',nb1)
        for i2 in range(nb22-1):
            ima_crop=ima[i1*dim:(i1+1)*dim,i2*dim+int(dim/2):(i2+1)*dim+int(dim/2),:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[i1*dim:(i1+1)*dim,i2*dim+int(3*dim/4):(i2+1)*dim+int(dim/4)]=predicted[:,int(dim/4):int(3*dim/4)]

    for i1 in range(nb21-1):
        print('4): ',i1,' / ',nb21-1)
        for i2 in range(nb22-1):
            ima_crop=ima[i1*dim+int(dim/2):(i1+1)*dim+int(dim/2),i2*dim+int(dim/2):(i2+1)*dim+int(dim/2),:]
            ima_crop=np.reshape(ima_crop,(1,dim,dim,bands))
            predicted = model1.predict(ima_crop,verbose=0)
            predicted = np.reshape(predicted,(dim,dim))
            pred_label[i1*dim+int(3*dim/4):(i1+1)*dim+int(dim/4),i2*dim+int(3*dim/4):(i2+1)*dim+int(dim/4)]=predicted[int(dim/4):int(3*dim/4),int(dim/4):int(3*dim/4)]
     
print('pred_image min  ',pred_label.min(),'  max ',pred_label.max())
print('imalin min  ',im_T.min(),'  max ',im_T.max())
print('im_GT min  ',im_GT.min(),'  max ',im_GT.max())

# pred_label=(pred_label*(im_T0.max()-im_T0.min()))+im_T0.min()
# pred_label=pred_label*max_val

# mx=np.max([pred_label.max(),im_T0.max(),im_GT.max()])
# pred_labeln=pred_label/mx
# ima1n=ima1/mx
# im_GTn=im_GT/mx


print('****** accuracies for linear upsampling******************')
RMSEl = rmse(im_T,im_GT,1) # mae(imagt255,predicted255,bands)
PSNRl=psnr(im_T,im_GT,max_val)
SSIMl = ssim(im_T,im_GT, data_range=100, multichannel=False)


print('eRMSEl = %.4f' % RMSEl)
print('PSNRl = %.3f' % PSNRl)
SSIM100l=SSIMl*100
print('SSIMl = %.3f' % SSIM100l)

print('****** accuracies for Unet SR ******************')

RMSE = rmse(pred_label,im_GT,1) # mae(imagt255,predicted255,bands)
PSNR=psnr(pred_label,im_GT,max_val)
SSIM = ssim(pred_label,im_GT, data_range=100, multichannel=False)


print('eRMSE = %.4f' % RMSE)
print('PSNR = %.3f' % PSNR)
SSIM100=SSIM*100
print('SSIM = %.3f' % SSIM100)

# profileT=im_T00.profile
# affine=profileT["transform"]
# px=affine[2]
# resolution=0.5
# py=affine[5]-(sz1-int(sz1*0.2))/resolution
# affine=[0.5,affine[1],affine[2],affine[3],-0.5,py]
# profileT.update(
#     width= sz2, 
#     height= sz1, 
#     transform= affine)

# # Graz Model
if(selected_model==1):
    with rasterio.open(cwd+'/prediction/Pred_Graz_Ts_TRGBEmissiv_Graz_mdl_LC08_190027_20210912_LST_C_small_aoi_2.tif', 'w', **profileT) as dst:
        dst.write(pred_label,1)
else:
    if(selected_model==2):
        # # Ferrara Model
        with rasterio.open(cwd+'/prediction/Pred_Graz_Ts_TRGBEmissiv_Ferrara_mdl_LC08_190027_20210912_LST_C_small_aoi_2.tif', 'w', **profileT) as dst:
            dst.write(pred_label,1)
    else:
        if(selected_model==3):

            # Graz_Ferrara Model
            with rasterio.open(cwd+'/prediction/Pred_Graz_Ts_TRGBEmissiv_Graz_Ferrara_mdl_LC08_190027_20210912_LST_C_small_aoi_2.tif', 'w', **profileT) as dst:
                dst.write(pred_label,1)


with rasterio.open(cwd+'/prediction/Linear_Graz_Ts_LC08_190027_20210912_LST_C_small_aoi_2.tif', 'w', **profileT) as dst:
    dst.write(im_T,1)

# with rasterio.open(cwd+'/prediction/Graz_day_Ts_surface_temperature_UTM33_mask_small_50cm_celcius_2.tif', 'w', **profileT) as dst:
#     dst.write(ima1org,1)        



    