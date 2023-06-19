#!/usr/bin/env python
#=============================================
# Copyright 2021 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases(DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
sys.path.append('../')
sys.path.append('/../../')
import numpy as np
from utils import image_utils
import nibabel as nib

def uncertainty_measures(data, label,pred_img,ras_cm):
    uncertainty = np.zeros((1, 2))

    #prevents log 0
    data = np.clip(data,0.00001, 1 - 0.00001)


    pred= np.argmax(data,axis=-1)
    new_pred_img= nib.MGHImage(pred, pred_img.affine, pred_img.header)
    new_pred_img=image_utils.clean_seg(new_pred_img, ras_cm)

    label_map=new_pred_img.get_fdata()
    temp_label=np.zeros_like(pred)

    if label in [1,2]:
        temp_label[label_map == label] = 1
    elif label == 3 :
        temp_label[label_map > 0] = 1

    # Entropy
    num = np.sum(temp_label)
    if num >0:
        uncertainty[0, 0]=( -1 * np.sum(temp_label*(data[:, :, :, 1] * np.log(data[:, :, :, 1])))/num)
    else:
        uncertainty[0,1]=np.nan

    uncertainty[0, 1] = num

    return uncertainty

def calculate_uncertainty_summary(matrix):
    uncertainty = np.zeros((1, 2))
    uncertainty[0,0] = np.nanmean(matrix[:,0])
    uncertainty[0,1] = np.nanstd(matrix[:,1]) /np.nanmean(matrix[:,1])
    return uncertainty


def calculate_uncertainty(logits,label,pred_img,cm):
    # Uncertainty
    sub_uncertainty = np.zeros((logits.shape[-1], 2))
    for i in range(logits.shape[-1]):
        sub_uncertainty[i, :] = uncertainty_measures(logits[:,:,:,:,i],label,pred_img,cm)

    uncertainty_values = calculate_uncertainty_summary(sub_uncertainty)

    return uncertainty_values



def extract_stats(prop,uncertain_values,SegID,Structname,col_names,voxel_size,zero_xyz):

    #col_names=['SubID','SegID','NVoxels','Volume_mm3','StructName','normMean','normMin','normMax','Entropy','CV','CM']


    metric_matrix=np.zeros((1,len(col_names)),dtype=object)

    #SegID
    metric_matrix[0, 1] = SegID
    #NVoxels
    metric_matrix[0, 2] = prop.area
    #Volume_mm3
    metric_matrix[0, 3] = np.around((prop.area * voxel_size),decimals=4)
    #StrucName
    metric_matrix[0, 4] = Structname
    #normMean
    metric_matrix[0, 5] = np.around(prop.mean_intensity,decimals=4)
    #normMin
    metric_matrix[0, 6] = np.around(prop.min_intensity,decimals=4)
    #normMax
    metric_matrix[0, 7] = np.around(prop.max_intensity,decimals=4)
    #Entropy
    metric_matrix[0, 8] = np.around(uncertain_values[0,0],decimals=4)
    #CV
    metric_matrix[0, 9] = np.around(uncertain_values[0,1],decimals=4)
    #CM
    centroid= prop.centroid
    centroid= np.array([centroid[0]+zero_xyz[0],centroid[1]+zero_xyz[1],centroid[2]+zero_xyz[2]]).astype(np.int16)

    metric_matrix[0, 10] = centroid

    return metric_matrix


def calculate_stats(args,save_dir,image,prediction,logits,cm,cm_logits,logger):
    import os
    import pandas as pd
    from skimage.measure import regionprops
    from scipy.ndimage.measurements import center_of_mass

    logger.info(30 * '-')
    logger.info('Calculating stats')

    voxel_size = np.prod(image.header.get_zooms())

    stats_dir = os.path.join(save_dir, 'stats')

    col_names=['SubID','SegID','NVoxels','Volume_mm3','StructName','normMean','normMin','normMax','Entropy','CV','CM']

    structures={'Left_OB':1,'Right_OB':2,'Total_OB':3}

    #initialized as nan everything that is not volume or area
    metrics_matrix = np.zeros((3,len(col_names)),dtype=object)
    metrics_matrix[:]=np.nan
    metrics_matrix[:,2]=0
    metrics_matrix[:,3]=0

    #initialized matrix values #SegID and StrucName
    for idx,struc in enumerate(structures.keys()):
        metrics_matrix[idx,1] = structures[struc]
        metrics_matrix[idx,4] = struc

    # compute label stats
    arr=np.array(prediction.get_fdata(),dtype=np.uint8)
    regions = regionprops(label_image=arr,intensity_image=image.get_fdata())
    unique_labels = np.unique(arr)

    #compute whole region stats
    arr_total=np.zeros_like(arr)
    arr_total[arr>0]=1
    total_region=regionprops(label_image=arr_total,intensity_image=image.get_fdata())


    #Warning flag
    if 1 not in unique_labels or 2 not in unique_labels:
        if 1 in unique_labels:
            logger.info('Warning check prediction map, OB from right hemisphere not found')
            warning_flag = 'OB from right hemisphere not found'
        elif 2 in unique_labels:
            logger.info('Warning check prediction map, OB from left hemisphere not found')
            warning_flag = 'OB from Left hemisphere not found'
        else:
            logger.info('Warning check prediction map, Case without an apparent OB')
            warning_flag = 'no apparent OB'
    else:
        warning_flag= None


    #Segmentation metrics
    for idx,struc in enumerate(structures.keys()):

        uncertain_values = calculate_uncertainty(logits, structures[struc], prediction, cm['ras'])
        # Entropy
        metrics_matrix[idx, 8] = np.around(uncertain_values[0, 0], decimals=4)
        # CV
        metrics_matrix[idx, 9] = np.around(uncertain_values[0, 1], decimals=4)

        if struc == 'Total_OB':
            no_labels = False
            try:
                prop = total_region[0]
            except:
                no_labels=True

            if not no_labels:
                metrics_matrix[idx, :] = extract_stats(prop, uncertain_values, SegID=structures[struc],
                                                       Structname=struc, col_names=col_names, voxel_size=voxel_size,
                                                       zero_xyz=cm['zero_xyz'])
        else:
            for prop in regions:
                if structures[struc] == prop.label:
                    metrics_matrix[idx, :] = extract_stats(prop, uncertain_values, SegID=structures[struc],
                                                           Structname=struc, col_names=col_names,voxel_size=voxel_size,zero_xyz=cm['zero_xyz'])


    metrics_matrix[:,0]=args.sub_id



    metrics_df = pd.DataFrame(metrics_matrix, columns=col_names)
    metrics_df.to_csv(os.path.join(stats_dir, 'segmentation_stats.csv'), sep=',', index=False)


    #localization metrics

    col_names=['SubID','loc_cm','seg_cm','dist_mm','mse_px','ROI_NVoxels','in_image']

    loc_matrix=np.zeros((1,len(col_names)),dtype=object)
    loc_matrix[:]=np.nan

    loc_matrix[0,0]=args.sub_id
    loc_matrix[0,1]=np.array([cm['xyz'][0],cm['xyz'][1],cm['xyz'][2]],dtype=(np.int16))
    if not no_labels:
        # calcualte cm of mass
        mass_center = center_of_mass(arr_total)
        mass_center = np.array([mass_center[0] + cm['zero_xyz'][0], mass_center[1] + cm['zero_xyz'][1],
                                mass_center[2] + cm['zero_xyz'][2]]).astype(np.int16)

        loc_matrix[0,2]=mass_center

        diff=loc_matrix[0,1]-loc_matrix[0,2]

        #dist_mm
        zooms=np.array(image.header.get_zooms())
        loc_matrix[0,3]=np.around(np.sum(np.square(diff)*zooms)**(1/2),decimals=4)
        #mse
        loc_matrix[0,4]=np.around(np.mean(np.square(diff)),decimals=4)

    #roi_voxels
    if np.any(cm_logits):
        loc = np.zeros_like(cm_logits)
        loc[cm_logits>0]=1
        loc_labels,count=np.unique(loc,return_counts=True)
        if 1 in loc_labels:
            loc_matrix[0,5]=count[1]
    else:
        loc_matrix[0, 5] = np.nan

    loc_matrix[0,6]=args.in_img.split('/')[-1].split('.')[0]
    #to-do warning flag for litte heat map maybe (1000) as sigma was 10 ,10**3
    loc_df = pd.DataFrame(loc_matrix, columns=col_names)
    loc_df.insert(len(col_names),'Flags',warning_flag)
    loc_df.to_csv(os.path.join(stats_dir, 'localization_stats.csv'), sep=',', index=False)


def calculate_stats_no_loc(args,save_dir):
    import os
    import pandas as pd
    from skimage.measure import regionprops
    from scipy.ndimage.measurements import center_of_mass


    stats_dir = os.path.join(save_dir, 'stats')

    #localization metrics

    col_names=['SubID','loc_cm','seg_cm','dist_mm','mse_px','ROI_NVoxels','in_image']

    loc_matrix=np.zeros((1,len(col_names)),dtype=object)
    loc_matrix[:]=np.nan

    warning_flag = 'Localization Network couldnt find OB region'

    loc_matrix[0,0]=args.sub_id
    loc_df = pd.DataFrame(loc_matrix, columns=col_names)
    loc_df.insert(len(col_names),'Flags',warning_flag)
    loc_df.to_csv(os.path.join(stats_dir, 'localization_stats.csv'), sep=',', index=False)


def obstats2table(main_dir,sub_list,name='obstats_table.csv'):
    from utils import misc
    import os
    import  pandas as pd
    loc_columns=['loc_cm','seg_cm','dist_mm','mse_px','ROI_NVoxels','in_image','Flags']

    metrics=['NVoxels','Volume_mm3','normMean','normMin','normMax','Entropy','CV','CM']

    structNames=['Left_OB','Right_OB','Total_OB']

    columns=[]

    for struc in structNames:
        for metric in metrics:
            columns.append(struc+ '_'+metric)

    columns.extend(loc_columns)

    subjects=[]

    table=np.zeros((len(sub_list),len(columns)),dtype=object)
    table[:]=np.nan


    for idx,sub in enumerate(sub_list):
        subjects.append(sub)
        seg_stats_file=misc.locate_file('*segmentation_stats.csv',os.path.join(main_dir,sub,'stats'))
        if seg_stats_file:
            df=pd.read_csv(seg_stats_file[0])
            idj =0
            for struc in structNames:
                stats=df.loc[df.StructName == struc]
                for metric in metrics:
                    table[idx,idj]=stats[metric].values[0]
                    idj +=1
        else:
            print('segmentation stat file(segmentation_stats.csv) not found in {}'.format(os.path.join(main_dir,sub,'stats')))
            idj = len(structNames) * len(metrics)

        loc_stats_file = misc.locate_file('*localization_stats.csv', os.path.join(main_dir, sub, 'stats'))
        df_loc=pd.read_csv(loc_stats_file[0])
        loc_values=df_loc.values
        table[idx,idj:] = loc_values[0,1:]


    final_table=table[:idx+1,:]

    df_table=pd.DataFrame(final_table,columns=columns)


    df_table.insert(loc=0, column='sub_id', value=subjects)

    df_table.to_csv(os.path.join(main_dir, name), sep=',', index=False)





