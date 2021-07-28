
# Copyright 2019 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases(DZNE)
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
import nibabel as nib
import scipy.ndimage
import os


def calculated_new_ornt(iornt,base_ornt):

    new_iornt=iornt[:]

    for axno, direction in np.asarray(base_ornt):
        idx=np.where(iornt[:,0] == axno)
        idirection=iornt[int(idx[0][0]),1]
        if direction == idirection:
            new_iornt[int(idx[0][0]), 1] = 1.0 #dont change view
        else:
            new_iornt[int(idx[0][0]), 1] = -1.0 #change view

    return new_iornt

def check_orientation(img,base_ornt=np.array([[0,-1],[1,1],[2,1]])):

    iornt=nib.io_orientation(img.affine)

    if not np.array_equal(iornt,base_ornt):
        img = img.as_reoriented(calculated_new_ornt(iornt,base_ornt))

    return img

def conform(img,flags,logger=None):
    """
    Args:
        img: nibabel img: Loaded source image
        flags: dict : Dictionary containing the image size, spacing and orientation
        order: int : interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    Returns:
        new_img: nibabel img : conformed nibabel image
    """
    # check orientation LAS
    img=check_orientation(img,base_ornt=flags['base_ornt'])

    img_arr=img.get_fdata()

    #Conform intensities
    src_min, scale = getscale(data=img_arr, dst_min=0, dst_max=255,logger=logger)
    img_arr = scalecrop(data=img_arr, dst_min=0, dst_max=255, src_min=src_min, scale=scale,logger=logger)

    new_img = nib.Nifti1Image(img_arr, img.affine, img.header)

    return new_img


def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999,logger=None):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.
    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: returns (adjusted) src_min and scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        sys.exit('ERROR: Min value in input is below 0.0!')

    if logger:
        logger.info("Input:    min: " + format(src_min) + "  max: " + format(src_max))
    else:
        print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        if logger:
            logger.info('ERROR: rescale upper bound not found')
        else:
            print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    if logger:
        logger.info("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))
    else:
        print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale


def scalecrop(data, dst_min, dst_max, src_min, scale,logger=None):
    """
    Function to crop the intensity ranges to specific min and max values
    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: scaled Image data array
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    if logger:
        logger.info("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))
    else:
     print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new