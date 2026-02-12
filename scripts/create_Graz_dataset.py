# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:09:34 2022

@author: 39388
"""

import numpy as np
import cv2
import math
# import matplotlib.pyplot as plt
from numpy.random import randn, rand
from numpy.random import randint, permutation
import os
import glob
import rasterio
from datetime import datetime
cwd = os.getcwd()
# cwd='/scratch/3DOM/smalek/building_old_maps/'
# dim=256
# dim=384
dim=512
cpTr=0
cpTs=0
cpVal=0


# # nbr of Tr of each image = 2000 = nbTr1*nbTr2 
# if(dim==256):
#     nbTr2=50  # number of Tr blocks horizontally 
#     nbTr1=80  # number of Tr blocks vertically
#     nbVal2=16  # number of Tr blocks horizontally 
#     nbVal1=16  # number of Tr blocks vertically
#     nbTs2=16  # number of Tr blocks horizontally 
#     nbTs1=32  # number of Tr blocks vertically

# if(dim==384):
#     nbTr2=50  # number of Tr blocks horizontally 
#     nbTr1=64  # number of Tr blocks vertically
#     nbVal2=10  # number of Tr blocks horizontally 
#     nbVal1=10  # number of Tr blocks vertically
#     nbTs2=10  # number of Tr blocks horizontally 
#     nbTs1=20  # number of Tr blocks vertically


# if(dim==512):
#     nbTr2=80  # number of Tr blocks horizontally 
#     nbTr1=50  # number of Tr blocks vertically
#     nbVal2=32  # number of Tr blocks horizontally 
#     nbVal1=2  # number of Tr blocks vertically
#     nbTs2=32  # number of Tr blocks horizontally 
#     nbTs1=4  # number of Tr blocks vertically
    
if(dim==512):
    nbTr2=10  # number of Tr blocks horizontally 
    nbTr1=8  # number of Tr blocks vertically
    nbVal2=8  # number of Tr blocks horizontally 
    nbVal1=2  # number of Tr blocks vertically
    nbTs2=8  # number of Tr blocks horizontally 
    nbTs1=4  # number of Tr blocks vertically    

im_T0=rasterio.open(cwd+'/Aligned_Graz_data/LC08_190027_20210912_LST_C_small_aoi_2_up05.tif')
im_GT0=rasterio.open(cwd+'/Aligned_Graz_data/Graz_day_surface_temperature_UTM33_mask_small_50cm_celcius_2.tif')
im_RGB05m0=rasterio.open(cwd+'/Aligned_Graz_data/RGB_mosaic_50cm_2.tif')
im_Emmiss05m0=rasterio.open(cwd+'/Aligned_Graz_data/graz_emissivity_50cm_masked_32633_2.tif')

im_T=im_T0.read(1)
im_GT=im_GT0.read(1)
im_B05m=im_RGB05m0.read(1)
im_G05m=im_RGB05m0.read(2)
im_R05m=im_RGB05m0.read(3)
im_Emmiss05m=im_Emmiss05m0.read(1)

sz1=im_GT.shape[0] # height
sz2=im_GT.shape[1] # width
print('im_GT shape=',im_GT.shape)

im_RGB05m=np.uint8(np.zeros((sz1,sz2,3)))
im_RGB05m[:,:,0]=im_R05m
im_RGB05m[:,:,1]=im_G05m
im_RGB05m[:,:,2]=im_B05m
# max_val=80

# im_GTn=im_GT/max_val
# im_Tn=im_T/max_val

# im_GTn=(im_GT-im_T.min())/(im_T.max()-im_T.min())
# im_Tn=(im_T-im_T.min())/(im_T.max()-im_T.min())

# im_GT=(im_GT-im_GT.min())/(im_GT.max()-im_GT.min())


# print('im_GT shape=',im_GT.shape)

dimTr=int(sz1*0.7) # 70% for Training data
posTr=0
stepTr1=int((dimTr-dim)/(nbTr1-1) - 1)  # steps vertically 
stepTr2=int((sz2-dim)/(nbTr2-1) - 1)  # steps horizontally  

dimVal=int(sz1*0.1) # 10% for validation data
posVal=dimTr
stepVal1=int((dimVal-dim)/(nbVal1-1) - 1) # steps vertically 
stepVal2=int((sz2-dim)/(nbVal2-1) - 1)  # steps horizontally 

dimTs=int(sz1*0.2) # 20% for [testing data
# dimTs1=int(sz1-dimVal)  # 2/3 of 30% = 20% (double of val area)

posTs=dimTr+dimVal
# posTs1=int(sz1/3)
stepTs1=int((dimTs-dim)/(nbTs1-1) - 1) # steps vertically 
stepTs2=int((sz2-dim)/(nbTs2-1)- 1)  # steps horizontally 
stp2=0
id1= -1
nbN=0
nbF1=0
nbF2=0
nbR=0
print('stepTs1=',stepTs1)
print('stepTs2=',stepTs2)
print('nbTs1=',nbTs1)
print('nbTs2=',nbTs2)
print('dimTs=',dimTs)
print('sz2=',sz2)
print('posTs=',posTs)



for i in range(nbTr2):
    stp1=0
    for j in range(nbTr1):
        cpTr=cpTr+1
        id1=(id1+1)%5
        print('cpTr=',cpTr)
        crop_RGB05m = im_RGB05m[stp1:stp1+dim,stp2:stp2+dim]
        crop_im_Emmiss05m = im_Emmiss05m[stp1:stp1+dim,stp2:stp2+dim]
        crop_GT = im_GT[stp1:stp1+dim,stp2:stp2+dim]
        crop_T = im_T[stp1:stp1+dim,stp2:stp2+dim]
        if(id1==0):
            nbF1=nbF1+1
            crop_RGB05m=cv2.flip(crop_RGB05m,1)
            crop_im_Emmiss05m=cv2.flip(crop_im_Emmiss05m,1)
            crop_GT=cv2.flip(crop_GT,1)
            crop_T=cv2.flip(crop_T,1)

            cv2.imwrite(cwd+'/Graz_dataset_512/Train/Input_RGB_05m/ima_'+str(cpTr)+'.png',crop_RGB05m)
            np.save(cwd+'/Graz_dataset_512/Train/Input_Emissivity_05m/ima_'+str(cpTr)+'.npy',crop_im_Emmiss05m)
            np.save(cwd+'/Graz_dataset_512/Train/Input_T_30m_up05/ima_'+str(cpTr)+'.npy',crop_T)
            np.save(cwd+'/Graz_dataset_512/Train/Output_T_05m/ima_'+str(cpTr)+'.npy',crop_GT) 
        else:
            if(id1==2):
                nbF2=nbF2+1
                crop_RGB05m=cv2.flip(crop_RGB05m,0)
                crop_im_Emmiss05m=cv2.flip(crop_im_Emmiss05m,0)
                crop_GT=cv2.flip(crop_GT,0)
                crop_T=cv2.flip(crop_T,0)

                cv2.imwrite(cwd+'/Graz_dataset_512/Train/Input_RGB_05m/ima_'+str(cpTr)+'.png',crop_RGB05m)
                np.save(cwd+'/Graz_dataset_512/Train/Input_Emissivity_05m/ima_'+str(cpTr)+'.npy',crop_im_Emmiss05m)
                np.save(cwd+'/Graz_dataset_512/Train/Input_T_30m_up05/ima_'+str(cpTr)+'.npy',crop_T)
                np.save(cwd+'/Graz_dataset_512/Train/Output_T_05m/ima_'+str(cpTr)+'.npy',crop_GT) 

            else:
                if(id1==4):
                    nbR=nbR+1
                    crop_RGB05m=np.rot90(crop_RGB05m)
                    crop_im_Emmiss05m=np.rot90(crop_im_Emmiss05m)
                    crop_GT=np.rot90(crop_GT)
                    crop_T=np.rot90(crop_T)

                    cv2.imwrite(cwd+'/Graz_dataset_512/Train/Input_RGB_05m/ima_'+str(cpTr)+'.png',crop_RGB05m)
                    np.save(cwd+'/Graz_dataset_512/Train/Input_Emissivity_05m/ima_'+str(cpTr)+'.npy',crop_im_Emmiss05m)
                    np.save(cwd+'/Graz_dataset_512/Train/Input_T_30m_up05/ima_'+str(cpTr)+'.npy',crop_T)
                    np.save(cwd+'/Graz_dataset_512/Train/Output_T_05m/ima_'+str(cpTr)+'.npy',crop_GT) 
                else:  
                    nbN=nbN+1  
                    cv2.imwrite(cwd+'/Graz_dataset_512/Train/Input_RGB_05m/ima_'+str(cpTr)+'.png',crop_RGB05m)
                    np.save(cwd+'/Graz_dataset_512/Train/Input_Emissivity_05m/ima_'+str(cpTr)+'.npy',crop_im_Emmiss05m)
                    np.save(cwd+'/Graz_dataset_512/Train/Input_T_30m_up05/ima_'+str(cpTr)+'.npy',crop_T)
                    np.save(cwd+'/Graz_dataset_512/Train/Output_T_05m/ima_'+str(cpTr)+'.npy',crop_GT) 

        # crop_RGB2 = cv2.resize(crop_RGB, dsize=(dim2, dim2), interpolation=cv2.INTER_LINEAR)
        # crop_DSM2 = cv2.resize(crop_DSM, dsize=(dim2, dim2), interpolation=cv2.INTER_LINEAR)
        # crop_GT2 = cv2.resize(crop_GT, dsize=(dim2, dim2), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(cwd+'/dataset512_3up2/Train/inputs_RGB/ima_'+str(cpTr)+'.png',crop_RGB2)
        # np.save(cwd+'/dataset512_3up2/Train/inputs_DSM/ima_'+str(cpTr)+'.npy',crop_DSM2)
        # cv2.imwrite(cwd+'/dataset512_3up2/Train/labels/ima_'+str(cpTr)+'.png',crop_GT2)
        stp1=stp1+stepTr1
    stp2=stp2+stepTr2
 
stp2=0
for i in range(nbVal2):
    stp1=0
    for j in range(nbVal1):
        cpVal=cpVal+1
        print('cpVal=',cpVal)
        crop_RGB05m = im_RGB05m[stp1+posVal:stp1+posVal+dim,stp2:stp2+dim]
        crop_im_Emmiss05m = im_Emmiss05m[stp1+posVal:stp1+posVal+dim,stp2:stp2+dim]        
        crop_GT = im_GT[stp1+posVal:stp1+posVal+dim,stp2:stp2+dim]
        crop_T = im_T[stp1+posVal:stp1+posVal+dim,stp2:stp2+dim]

        cv2.imwrite(cwd+'/Graz_dataset_512/Val/Input_RGB_05m/ima_'+str(cpVal)+'.png',crop_RGB05m)
        np.save(cwd+'/Graz_dataset_512/Val/Input_Emissivity_05m/ima_'+str(cpVal)+'.npy',crop_im_Emmiss05m)
        np.save(cwd+'/Graz_dataset_512/Val/Input_T_30m_up05/ima_'+str(cpVal)+'.npy',crop_T)
        np.save(cwd+'/Graz_dataset_512/Val/Output_T_05m/ima_'+str(cpVal)+'.npy',crop_GT)

        # crop_RGB2 = cv2.resize(crop_RGB, dsize=(dim2, dim2), interpolation=cv2.INTER_LINEAR)
        # crop_DSM2 = cv2.resize(crop_DSM, dsize=(dim2, dim2), interpolation=cv2.INTER_LINEAR)
        # crop_GT2 = cv2.resize(crop_GT, dsize=(dim2, dim2), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(cwd+'/dataset512_3up2/Val/inputs_RGB/ima_'+str(cpVal)+'.png',crop_RGB2)
        # np.save(cwd+'/dataset512_3up2/Val/inputs_DSM/ima_'+str(cpVal)+'.npy',crop_DSM2)
        # cv2.imwrite(cwd+'/dataset512_3up2/Val/labels/ima_'+str(cpVal)+'.png',crop_GT2)
        stp1=stp1+stepVal1
    stp2=stp2+stepVal2        

stp2=0        
for i in range(nbTs2):
    stp1=0
    for j in range(nbTs1):
        cpTs=cpTs+1
        print('cpTs=',cpTs)
        crop_RGB05m = im_RGB05m[stp1+posTs:stp1+posTs+dim,stp2:stp2+dim]
        crop_im_Emmiss05m = im_Emmiss05m[stp1+posTs:stp1+posTs+dim,stp2:stp2+dim]        
        crop_GT = im_GT[stp1+posTs:stp1+posTs+dim,stp2:stp2+dim]
        crop_T = im_T[stp1+posTs:stp1+posTs+dim,stp2:stp2+dim]

        cv2.imwrite(cwd+'/Graz_dataset_512/Test/Input_RGB_05m/ima_'+str(cpTs)+'.png',crop_RGB05m)
        np.save(cwd+'/Graz_dataset_512/Test/Input_Emissivity_05m/ima_'+str(cpTs)+'.npy',crop_im_Emmiss05m)
        np.save(cwd+'/Graz_dataset_512/Test/Input_T_30m_up05/ima_'+str(cpTs)+'.npy',crop_T)
        np.save(cwd+'/Graz_dataset_512/Test/Output_T_05m/ima_'+str(cpTs)+'.npy',crop_GT)

        # crop_RGB2 = cv2.resize(crop_RGB, dsize=(dim2, dim2), interpolation=cv2.INTER_LINEAR)
        # crop_DSM2 = cv2.resize(crop_DSM, dsize=(dim2, dim2), interpolation=cv2.INTER_LINEAR)
        # crop_GT2 = cv2.resize(crop_GT, dsize=(dim2, dim2), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(cwd+'/dataset512_3up2/Test/inputs_RGB/ima_'+str(cpTs)+'.png',crop_RGB2)
        # np.save(cwd+'/dataset512_3up2/Test/inputs_DSM/ima_'+str(cpTs)+'.npy',crop_DSM2)
        # cv2.imwrite(cwd+'/dataset512_3up2/Test/labels/ima_'+str(cpTs)+'.png',crop_GT2)
        stp1=stp1+stepTs1
    stp2=stp2+stepTs2    

print('Nr=',cpTr)        
print('nbN=',nbN)        
print('nbF1=',nbF1)
print('nbF2=',nbF2)
print('nbR=',nbR)
print('Nv=',cpVal)        
print('Ns=',cpTs)        
        
