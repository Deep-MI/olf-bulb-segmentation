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
sys.path.append('../../')
import torch
import torch.nn as nn
import numpy as np
import time
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils.datasetUtils import testDataset
from utils.transformUtils import ToTensorTest
from utils.image_utils import plane_swap, map_size , get_thick_slices, clean_seg
from utils import misc as misc
from scipy.special import softmax
import os




def select_model(arc,modelConfig):
    from models.FastSurferCNN import FastSurferCNN
    from models.AttFastSurferCNN import AttFastSurferCNN

    if arc == 'FastSurferCNN':
     return FastSurferCNN(modelConfig.copy())
    elif arc == 'AttFastSurferCNN':
        return AttFastSurferCNN(modelConfig.copy())
    else:
        raise ValueError("Model {} not found".format(arc))

def select_plane(value,planes):
    for plane in planes:
        if str(plane) in str(value):
            return plane

class OBNet(object):

    seg_params_network = {'num_channels': 3, 'num_filters': 64,
                      'kernel_h': 3, 'kernel_w': 3, 'stride_conv': 1,
                      'pool': 2, 'stride_pool': 2, 'num_classes': 2,
                      'kernel_c': 1, 'kernel_d': 3,  'dilation' : 1}

    loc_params_network = {'num_channels': 3, 'num_filters': 64,
                          'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1,
                          'pool': 2, 'stride_pool': 2, 'num_classes': 2,
                          'kernel_c': 1, 'kernel_d': 1, 'dilation': 1}



    def __init__(self,args,flags,logger):

        self.args = args
        self.flags = flags
        self.logger=logger
        self.seg_params_network = OBNet.seg_params_network.copy()
        self.loc_params_network = OBNet.loc_params_network.copy()
        self.device,self.use_cuda,self.model_parallel=self.check_device()

    def check_device(self):
        # Put it onto the GPU or CPU
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.logger.info("Cuda available: {}, "
              "# Available GPUS: {}, "
              "Cuda user disabled (--no_cuda flag): {}, "
              "--> Using device: {}".format(torch.cuda.is_available(), torch.cuda.device_count(), self.args.no_cuda, device))

        if torch.cuda.device_count() > 1 and not self.args.no_cuda:
            model_parallel = True
        else:
            model_parallel = False

        return device,use_cuda,model_parallel

    def load_weights(self,current_model):
        from collections import OrderedDict

        model_state = torch.load(current_model, map_location=self.device)

        new_state_dict = OrderedDict()

        for k, v in model_state["model_state_dict"].items():

            if k[:7] == "module." and not self.model_parallel:
                new_state_dict[k[7:]] = v

            elif k[:7] != "module." and self.model_parallel:
                new_state_dict["module." + k] = v

            else:
                new_state_dict[k] = v

        return new_state_dict

    def predict(self,img,batch_size,model):
        transform_test = transforms.Compose([ToTensorTest()])
        test_dataset = testDataset(img, transforms=transform_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        pred_logits = []

        model.eval()
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(test_loader):
                print('---> Procesing batch {}  from {}'.format(batch_idx+1,len(test_loader)))
                images_batch = sample_batch['image']

                if self.use_cuda:
                    images_batch = images_batch.cuda()

                temp = model(images_batch)
                pred_logits.append(temp.cpu())

            pred_logits = torch.cat(pred_logits, axis=0)

        # change from N,C,W,H to view with C in last dimension = N,W,H,C
        pred_logits = pred_logits.permute(0, 2, 3, 1)
        pred_logits = pred_logits.numpy()

        del images_batch
        return pred_logits

    def run_localization(self,t2_img):
        import nibabel.processing
        from scipy.ndimage.measurements import center_of_mass

        model = select_model(self.flags['loc_arc'],self.loc_params_network.copy())

        if self.model_parallel:
            model = nn.DataParallel(model)

        model.to(self.device)


        resampled_img = nibabel.processing.resample_to_output(t2_img, self.flags['localization']['spacing'],order=1)

        orig_arr= resampled_img.get_fdata()
        orig_shape = orig_arr.shape

        planes = ['axial','coronal', 'sagittal']
        sub_logits=[]

        start_loc=time.time()

        for key,seg_loc in self.flags['localization']['models'].items():
            plane = select_plane(seg_loc,planes)
            self.logger.info("--->Testing {} localization model".format(plane))
            # load model
            model_state = self.load_weights(seg_loc)
            model.load_state_dict(model_state)
            self.logger.info('Model weights loaded from {}'.format(seg_loc))
            # organize data
            self.logger.info('Input data shape {}'.format(orig_shape))
            mod_arr = plane_swap(orig_arr, plane=plane)
            mod_arr = map_size(mod_arr, base_shape=(
            mod_arr.shape[0], self.flags['localization']['imgSize'][0], self.flags['localization']['imgSize'][1]), verbose=0)
            mod_arr = get_thick_slices(mod_arr,self.flags['thickness'])
            self.logger.info('input data transform to {}'.format(mod_arr.shape))


            start_model = time.time()

            # evaluate
            logits = self.predict(mod_arr, batch_size=self.flags['batch_size'], model=model)

            #organice data
            logits = plane_swap(logits,plane,inverse=True)
            # remove padding
            temp_logits = np.zeros((orig_shape[0], orig_shape[1], orig_shape[2], logits.shape[3]))
            for i in range(logits.shape[3]):
                temp_logits[:, :, :, i] = map_size(logits[:, :, :, i],
                                                   base_shape=(orig_shape[0], orig_shape[1], orig_shape[2]), verbose=0)

            logits = softmax(temp_logits, axis=-1)
            self.logger.info('output data shape {}'.format(logits.shape))
            logits = logits[..., np.newaxis]

            sub_logits.append(logits)

            model_end = time.time() - start_model
            self.logger.info("Model Done in {:0.4f} seconds".format(model_end))

        sub_logits = np.concatenate(sub_logits, axis=-1)
        sub_arr = np.sum(sub_logits, axis=-1) / len(planes)
        pred_arr = np.argmax(sub_arr, axis=-1)

        loc_end = time.time() - start_loc
        self.logger.info("---> Finish localization models in {:0.4f} seconds".format(loc_end))

        try:
            pred_cm = np.array(center_of_mass(pred_arr), dtype=np.int16)

        except:
            self.logger.info('ERROR: localization network cannot detect the region of interest. Please check image quality')
            pred_cm = None

        del model

        return pred_cm,sub_arr[:,:,:,1],resampled_img


    def run_segmentation(self,t2_arr):

        model = select_model(self.flags['seg_arc'],self.seg_params_network.copy())

        num_classes=self.seg_params_network['num_classes']

        if self.model_parallel:
            model = nn.DataParallel(model)

        model.to(self.device)

        planes = ['axial','coronal','sagittal']

        orig_shape = t2_arr.shape

        sub_logits = np.ndarray((orig_shape[0], orig_shape[1], orig_shape[2],num_classes,0), dtype=np.float32)

        start_seg=time.time()

        for key,seg_model in self.flags['segmentation']['models'].items():
            plane = select_plane(seg_model,planes)
            self.logger.info("--->Testing {} segmentation model".format(plane))
            #load model
            model_state = self.load_weights(seg_model)
            model.load_state_dict(model_state)
            self.logger.info('Model weights loaded from {}'.format(seg_model))

            # organize data
            mod_arr = plane_swap(t2_arr, plane=plane)
            mod_arr = map_size(mod_arr, base_shape=(mod_arr.shape[0], self.flags['segmentation']['imgSize'][0],
                                                    self.flags['segmentation']['imgSize'][1]),verbose=0)

            start_model = time.time()

            mod_arr = get_thick_slices(mod_arr, self.flags['thickness'])
            logits = self.predict(mod_arr, batch_size=self.flags['batch_size'], model=model)

            # organice data
            logits = plane_swap(logits, plane, inverse=True)
            # remove padding
            temp_logits = np.zeros((orig_shape[0], orig_shape[1], orig_shape[2], logits.shape[3]))
            for i in range(logits.shape[3]):
                temp_logits[:, :, :, i] = map_size(logits[:, :, :, i],
                                                   base_shape=(orig_shape[0], orig_shape[1], orig_shape[2]),
                                                   verbose=0)

            logits = softmax(temp_logits, axis=-1)

            logits = logits[..., np.newaxis]

            sub_logits=np.append(sub_logits,logits,axis=-1)

            end_model = time.time() - start_model
            self.logger.info("Model Done in {:0.4f} seconds".format(end_model))

        sub_arr = np.sum(sub_logits, axis=-1)
        pred_arr = np.argmax(sub_arr, axis=-1)

        end_seg = time.time() - start_seg
        self.logger.info("---> Finish segmentation models in {:0.4f} seconds".format(end_seg))

        del model
        return pred_arr,sub_logits


    def eval(self, t2_img,save_dir):
        import nibabel as nib
        import nibabel.processing
        import h5py

        mri_folder=os.path.join(save_dir,'mri')
        misc.create_exp_directory(mri_folder)

        i_zoom = t2_img.header.get_zooms()
        if self.args.no_interpolate:
            self.logger.info('T2 image will be process at native image resolution({}).\n'
                             'Warning: The pipeline was validated in T2 scans with an input resolution of 0.7 and 0.8 mm isotropic.\n'
                             'T2 scans with another resolution is highly recommended that segmentation quality is assessed by the user or re-run the pipeline using the default mode.'.format(i_zoom))
        else:
            if not np.allclose(np.array(i_zoom), np.array(self.flags['spacing']), rtol=0.05):
                self.logger.info('Interpolating image from resolution {} to {}'.format(i_zoom,self.flags['spacing']))
                t2_img = nibabel.processing.resample_to_output(t2_img, self.flags['spacing'],order=self.args.order)

        #Save pipeline input Image
        t2_img.set_data_dtype(np.uint8)
        nib.save(t2_img,os.path.join(mri_folder,'orig.nii.gz'))

        if not self.args.no_localization:
            t2_arr = t2_img.get_fdata()
            self.logger.info(30 * '-')
            self.logger.info('Running localization models')

            t2_cm, cm_logits, resampled_img= self.run_localization(t2_img)

            if np.any(t2_cm):
                cm_logits[cm_logits<0.5]= 0

                resampled_img.set_data_dtype(np.uint8)
                nib.save(resampled_img,os.path.join(mri_folder,'loc_orig.nii.gz'))

                loc_pred = nib.Nifti1Image(cm_logits, resampled_img.affine, resampled_img.header)
                loc_pred.set_data_dtype(np.float32)
                nib.save(loc_pred, os.path.join(mri_folder, 'loc_heatmap.nii.gz'))

                #transform coordinates
                coord = np.array((t2_cm[0], t2_cm[1], t2_cm[2], 1))

                orig_coord={}

                orig_coord['ras'] = np.dot(resampled_img.affine, coord)

                orig_coord['xyz'] = np.dot(np.linalg.inv(t2_img.affine), orig_coord['ras'])
                orig_coord['xyz'] = orig_coord['xyz'].astype(np.int16)

                self.logger.info(30 * '-')
                self.logger.info('Crop image from coordinate % d, %d , %d' %(orig_coord['xyz'][0],orig_coord['xyz'][1],orig_coord['xyz'][2]))
                self.logger.info(30 * '-')

                padding=self.flags['segmentation']['imgSize'][0] // 2
                zero_coordinate = np.array([orig_coord['xyz'][0] - padding, orig_coord['xyz'][1] - padding,
                                            orig_coord['xyz'][2] - padding, 1])

                orig_coord['zero_xyz']=zero_coordinate

                crop_t2_arr = t2_arr[orig_coord['xyz'][0] - padding:orig_coord['xyz'][0] + padding,
                             orig_coord['xyz'][1] - padding:orig_coord['xyz'][1] + padding,
                             orig_coord['xyz'][2] - padding:orig_coord['xyz'][2] + padding]


                self.logger.info(30 * '-')
                self.logger.info('zero coordinate')
                self.logger.info(zero_coordinate)
                translationRAS = np.dot(t2_img.affine, zero_coordinate)

                vox2RAS = t2_img.affine[:]

                vox2RAS[0, 3] = translationRAS[0]
                vox2RAS[1, 3] = translationRAS[1]
                vox2RAS[2, 3] = translationRAS[2]

                crop_t2=nib.Nifti1Image(crop_t2_arr,vox2RAS, t2_img.header)

            else:
                return None, None, None, None, None

        else:
            crop_t2_arr = t2_img.get_fdata()
            #middle point from crop image

            coord = np.array((crop_t2_arr.shape[0] // 2, crop_t2_arr.shape[1] // 2, crop_t2_arr.shape[2] // 2, 1))
            crop_t2_arr = map_size(crop_t2_arr,base_shape=(self.flags['segmentation']['imgSize'][0],self.flags['segmentation']['imgSize'][1],self.flags['segmentation']['imgSize'][0]))
            #crop_t2 =  nib.Nifti1Image(crop_t2_arr,t2_img.affine,t2_img.header)
            orig_coord = {}
            orig_coord['ras'] = np.dot(t2_img.affine, coord)
            orig_coord['xyz'] = coord[:]
            orig_coord['xyz'] = orig_coord['xyz'].astype(np.int16)

            zero_coordinate = np.array([orig_coord['xyz'][0] - crop_t2_arr.shape[0] //2, orig_coord['xyz'][1] - crop_t2_arr.shape[1] //2,
                                        orig_coord['xyz'][2] - crop_t2_arr.shape[2] //2, 1])

            orig_coord['zero_xyz']=zero_coordinate

            self.logger.info(30 * '-')
            self.logger.info('zero coordinate')
            self.logger.info(zero_coordinate)
            translationRAS = np.dot(t2_img.affine, zero_coordinate)

            vox2RAS = t2_img.affine[:]

            vox2RAS[0, 3] = translationRAS[0]
            vox2RAS[1, 3] = translationRAS[1]
            vox2RAS[2, 3] = translationRAS[2]

            crop_t2 = nib.Nifti1Image(crop_t2_arr, vox2RAS, t2_img.header)

            cm_logits = None


        # ----------Segmentation----------
        self.logger.info(30 * '-')
        self.logger.info('Running segmentation models')

        prediction, logits = self.run_segmentation(crop_t2_arr)

        pred_img = nib.Nifti1Image(prediction, crop_t2.affine, crop_t2.header)
        pred_img=clean_seg(pred_img,orig_coord['ras'])

        if self.args.orig_res:
            if not self.args.no_interpolate:
                self.logger.info('Interpolating prediction from resolution {} to {}'.format(t2_img.header.get_zooms(), i_zoom))
                pred_img = nibabel.processing.resample_to_output(pred_img, i_zoom, order=0)
                crop_t2 = nibabel.processing.resample_to_output(crop_t2, i_zoom, order=self.args.order)
                crop_t2.set_data_dtype(np.uint8)
                nib.save(crop_t2, os.path.join(mri_folder, 'orig_crop.nii.gz'))
            else:
                self.logger.info('Segmentation map is already @ input image native resolution {}'.format(i_zoom))
                self.logger.info('No interpolation need it as no_interpolate flag is {}'.format(self.args.no_interpolate))

        pred_img.set_data_dtype(np.int16)
        nib.save(pred_img,os.path.join(mri_folder,'ob_seg.nii.gz'))

        if self.args.save_logits:
            logit_file=os.path.join(mri_folder,'ob_seg_logits.h5')
            hf = h5py.File(logit_file, 'w')
            hf.create_dataset('Data', data=logits.astype(np.float32), compression='gzip')
            hf.close()
        return pred_img,crop_t2, logits ,orig_coord,cm_logits

