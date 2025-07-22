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

import numpy as np

def plane_swap(data,plane,inverse=False):
    if plane == 'axial':
        if not inverse:
            return np.moveaxis(data,[0,1,2],[2,1,0])
        else:
            return np.moveaxis(data,[0,1,2],[2,1,0])

    elif plane == 'coronal':
        if not inverse:
            return np.moveaxis(data,[0,1,2],[2,0,1])
        else:
            return np.moveaxis(data,[0, 1, 2],[1,2,0])
    elif plane == 'sagittal':
        return data


def define_size(mov_dim,ref_dim):
    """Calculate a new image size by duplicate the size of the bigger ones
    Args:
        move_dim (3D array sise):  3D size of the input volume
        ref_dim (3D ref size) : 3D size of the reference size
    Returns:
        new_dim (list) : New array size
        borders (list) : border Index for mapping the old volume into the new one
    """
    new_dim=np.zeros(len(mov_dim),dtype=int)
    borders=np.zeros((len(mov_dim),2),dtype=int)

    padd = [int(mov_dim[0] // 2), int(mov_dim[1] // 2), int(mov_dim[2] // 2)]

    for i in range(len(mov_dim)):
        new_dim[i]=int(max(2*mov_dim[i],2*ref_dim[i]))
        borders[i,0]= int(new_dim[i] // 2) -padd [i]
        borders[i,1]= borders[i,0] +mov_dim[i]

    return list(new_dim),borders

def map_size(arr,base_shape,verbose=1):
    """Padd or crop the size of an input volume to a reference shape
    Args:
        arr (3D array array):  array to be map
        base_shape (3D ref size) : 3D size of the reference size
    Returns:
        final_arr (3D array) : 3D array containing with a shape defined by base_shape
    """
    if verbose >0:
        print('Volume will be resize from %s to %s ' % (arr.shape, base_shape))

    new_shape,borders=define_size(np.array(arr.shape),np.array(base_shape))
    new_arr=np.zeros(new_shape)
    final_arr=np.zeros(base_shape)

    new_arr[borders[0,0]:borders[0,1],borders[1,0]:borders[1,1],borders[2,0]:borders[2,1]]= arr[:]

    middle_point = [int(new_arr.shape[0] // 2), int(new_arr.shape[1] // 2), int(new_arr.shape[2] // 2)]
    padd = [int(base_shape[0]/2), int(base_shape[1]/2), int(base_shape[2]/2)]

    low_border=np.array((np.array(middle_point)-np.array(padd)),dtype=int)
    high_border=np.array(np.array(low_border)+np.array(base_shape),dtype=int)

    final_arr[:,:,:]= new_arr[low_border[0]:high_border[0],
                   low_border[1]:high_border[1],
                   low_border[2]:high_border[2]]

    return final_arr


def get_thick_slices(img_data, slice_thickness=3):
    """
    Function to extract thick slices from the image
    (feed slice_thickness preceeding and suceeding slices to network,
    label only middle one)
    :param np.ndarray img_data: 3D MRI image read in with nibabel
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3)
    :return:
    """
    d ,h, w  = img_data.shape
    img_data_pad = np.expand_dims(np.pad(img_data, ((slice_thickness, slice_thickness),(0, 0), (0, 0)), mode='edge'),
                                  axis=3)
    img_data_thick = np.ndarray((d, h, w, 0), dtype=np.uint8)

    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[slice_idx:d + slice_idx,:, :, :], axis=3)

    return img_data_thick


def clean_seg(label_img,ras_cm):
    from skimage.measure import label, regionprops
    import nibabel as nib


    arr = label(label_img.get_fdata())

    labels, count = np.unique(arr, return_counts=True)

    if len(labels)>1:

        regions = regionprops(arr)

        ort = nib.aff2axcodes(label_img.affine)

        new_pred=np.zeros(arr.shape)

        for prop in regions:
            coord=prop.centroid
            coord= np.array((coord[0], coord[1], coord[2], 1))
            ras_coord = np.dot(label_img.affine, coord)

            # TODO: Fix left and right issues if localization network fails CM is different

            if ort[0] == 'L':

                if ras_coord[0] > ras_cm[0]:
                    new_pred[arr ==prop.label] = 2
                else:
                    new_pred[arr ==prop.label] = 1

            elif ort[0] == 'R':

                if ras_coord[0] < ras_cm[0]:
                    new_pred[arr ==prop.label] = 1
                else:
                    new_pred[arr ==prop.label] = 2

        return nib.MGHImage(new_pred, label_img.affine, label_img.header)

    else:
        return label_img


