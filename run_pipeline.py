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
import numpy as np
import os
import time
import argparse
from utils import misc as misc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def read_config(path):
    import yaml
    print("Given configuration file: ", path)
    with open(path, 'r') as file:
        configStruct = yaml.load(file,Loader=yaml.FullLoader)
    return configStruct

def get_full_paths(weights_dict,root_dir):
    new_dict={}
    for key,value in weights_dict.items():
        path = misc.locate_file(value,root_dir)

        if os.path.isfile(path[0]):
            new_dict[key]=path[0]
        else:
            raise ValueError('File for model {} doesnt exist : {}'.format(key,value))

    return  new_dict


def ob_pipeline(args,flags):
    import nibabel as nib
    from utils import conform as conform
    from models.OBNet import OBNet
    from utils import stats
    from utils import visualization

    save_dir=os.path.join(args.output_dir,args.sub_id)
    misc.create_exp_directory(save_dir)
    logger = misc.setup_logger(os.path.join(save_dir, "log.txt"))

    start = time.time()

    if os.path.isfile(args.in_img):

        logger.info('Reading file {}'.format(args.in_img))
        #load t2 image
        t2_orig_img=nib.load(args.in_img)
        #Conform Image intensities to 0 to 255
        t2_img=conform.conform(t2_orig_img,flags,logger)

        #Prediction
        pipeline= OBNet(args,flags,logger)

        pred_img,t2_crop, logits,coords,cm_logits= pipeline.eval(t2_img,save_dir)

        misc.create_exp_directory(os.path.join(save_dir, 'stats'))

        #Compute Segmentation Stats
        if coords:
            misc.create_exp_directory(os.path.join(save_dir,'QC'))

            visualization.plot_qc_images(save_dir=save_dir,image=t2_crop,prediction=pred_img)


            stats.calculate_stats(args,save_dir,image=t2_crop,prediction=pred_img,logits=logits,cm=coords,cm_logits=cm_logits,logger=logger)

            end = time.time() - start

            logger.info("Total computation time :  %0.4f seconds." % end)
            logger.info('\n')
            logger.info('Thank you for using the automated olfactory bulb segmentation pipeline')
            logger.info('If you find it useful and use it for a publication, please cite: ')
            logger.info('Estrada, Santiago, et al. "Automated Olfactory Bulb Segmentation on High Resolutional T2-Weighted MRI."\n'
                        'NeuroImage (2021): 118464. https://doi.org/10.1016/j.neuroimage.2021.118464')
        else:
            stats.calculate_stats_no_loc(args, save_dir)

    else:
        logger.info('ERROR: file {} not found'.format(args.in_img))



def option_parse():
    parser = argparse.ArgumentParser(
        description='Olfactory bulb segmentation pipeline for high resolutional T2-weighted MRI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-in", "--in_img", help="T2 image path", required=True)

    parser.add_argument("-out", "--output_dir", help="Main output directory where pipeline results are going to be store", required=True)

    parser.add_argument("-sid", "--sub_id", type=str, help="subject id", required=True,
                        default='subid')

    parser.add_argument('-batch', "--batch_size", type=int,
                        help='Batch size for inference by default is 8', required=False, default=8)

    parser.add_argument('-gpu', "--gpu_id", type=int,
                        help='GPU device name to run model', required=False, default=0)
    parser.add_argument('-ncuda', "--no_cuda", action='store_true',
                        help='Disable CUDA (no GPU usage, inference on CPU)', required=False)
    parser.add_argument('-ninter', "--no_interpolate", action='store_true',
                        help='No interpolate input scans to the default training resolution of 0.8mm isotropic', required=False)
    parser.add_argument('-order', "--order", type=int,
                        help='interpolation order to used if input scan is interpolated (0=nearest,1=linear(default),2=quadratic,3=cubic)', required=False, default=1)

    parser.add_argument('-logits', "--save_logits", action='store_true',
                        help='Save segmentation logits maps as a h5 file', required=False)

    parser.add_argument('-model', "--model", type=int,
                        help='AttFastSurferCNN model to be run by default the pipeline runs all 4 AttFastSurferCNN models;\n'
                             '(1 = model 1,2 = model 2,3 = model 3, 4 = model 4, 5= all models(default))', required=False, default=5)

    parser.add_argument('-ores', '--orig_res', action='store_true', help='Upsample or downsample OB segmentation to the input image resolution;\n'
                                                                     ' by default the pipeline produces a segmentation with a 0.8mm isotropic resolution', required=False)

    parser.add_argument('-loc_dir','--loc_dir',help='Localization weights directory',required=False,default='./LocModels')

    parser.add_argument('-seg_dir','--seg_dir',help='Segmentation weights directory',required=False,default='./SegModels')

    args = parser.parse_args()

    FLAGS=set_up_model(seg_dir=args.seg_dir,seg_arc='AttFastSurferCNN',loc_dir=args.loc_dir,loc_arc='FastSurferCNN',model=args.model)

    FLAGS.update({'batch_size':args.batch_size})
    FLAGS.update({'loc_arc': 'FastSurferCNN' })
    FLAGS.update({'seg_arc': 'AttFastSurferCNN'})

    return args,FLAGS


def set_up_model(seg_dir,seg_arc,loc_dir,loc_arc,model):

    FLAGS = {}

    FLAGS['base_ornt'] = np.array([[0, -1], [1, 1], [2, 1]])
    FLAGS['spacing'] = [float(0.8), float(0.8), float(0.8)]

    FLAGS['batch_size'] = 8
    FLAGS['thickness'] = 1

    FLAGS['num_classes'] = 2

    # Segmenation model
    FLAGS['segmentation'] = {}

    FLAGS['segmentation']['imgSize'] = [96, 96]
    FLAGS['segmentation']['models'] = {}

    FLAGS['localization'] = {}
    FLAGS['localization']['imgSize'] = [192, 192]
    FLAGS['localization']['spacing'] = [1.6, 1.6, 1.6]
    FLAGS['localization']['models'] = {}

    seg_config = os.path.join(seg_dir,seg_arc, seg_arc+'_weights.yml')
    seg_weights = read_config(seg_config)

    if model in [1,2,3,4,5]:
        if model != 5:
            aux_weights={}
            split = 'split_'+ str(model)
            for weight in seg_weights:
                if split in seg_weights[weight]:
                    aux_weights[weight]=seg_weights[weight]
            del seg_weights
            seg_weights = aux_weights.copy()
    else:
        print('Model {} option not available, model option will be change to all models'.format(model))

    seg_weights = get_full_paths(seg_weights, seg_dir)
    FLAGS['segmentation']['models'].update(seg_weights)

    loc_config = os.path.join(loc_dir,loc_arc, loc_arc+'_weights.yml')
    loc_weights = read_config(loc_config)
    loc_weights = get_full_paths(loc_weights, loc_dir)

    FLAGS['localization']['models'].update(loc_weights)

    return FLAGS

if __name__ == '__main__':

    args,FLAGS= option_parse()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id);

    ob_pipeline(args,FLAGS)

    sys.exit(0)



