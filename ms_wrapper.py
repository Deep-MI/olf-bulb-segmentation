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


import subprocess
import shlex
import sys
import argparse
sys.path.append('../')
sys.path.append('../../')

from utils import stats


def run_cmd(cmd):
    """
    execute the comand
    """
    print('#@# Command: ' + cmd + '\n')
    args = shlex.split(cmd)
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        print('ERROR: ' + 'cannot run command')
        # sys.exit(1)
        raise
    print('\n')



def option_parse():
    parser = argparse.ArgumentParser(
        description='Wrapper for running the olfactory bulb pipeline in multiple scans',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-slist", "--sublist", help="Subject List", required=True)

    parser.add_argument("-indir", "--data_dir", type=str, help="data directory", required=True)

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

    return args


def check_paths(sublist,root_dir):
    import pandas as pd
    import os
    import numpy as np
    from ob_pipeline.utils import misc

    df=pd.read_csv(sublist,sep=',')

    arr=df.values

    new_arr=np.zeros(arr.shape,dtype=object)

    idx=0

    for i in range(arr.shape[0]):
        sub=str(arr[i,0])
        t2_prefix=arr[i,1]

        t2_path=misc.locate_file('*'+t2_prefix,os.path.join(root_dir,sub))

        if t2_path:
            if os.path.isfile(t2_path[0]):
                new_arr[idx,0]=sub
                new_arr[idx,1]=t2_path[0]
                idx += 1
            else:
                print('ERROR: path {} is not a file '.format(t2_path[0]))
        else:
            print('ERROR image {} not found at directory {}'.format(t2_prefix,os.path.join(root_dir,sub)))


    return new_arr[:idx,:]


if __name__=='__main__':

    args= option_parse()

    sub_list=check_paths(args.sublist,args.data_dir)

    for i in range(sub_list.shape[0]):
        cmd='python3 ./run_pipeline.py -in {} -out {} -sid {} -batch {} -gpu {} ' \
            '-loc_dir {}  -seg_dir {}  -model {} -order {}'.format(sub_list[i,1],args.output_dir,
                                                                    sub_list[i,0],args.batch_size,
                                                                    args.gpu_id,args.loc_dir,
                                                                    args.seg_dir,args.model,args.order)

        if args.no_cuda:
            cmd =cmd +' -ncuda'
        if args.save_logits:
            cmd = cmd + ' -logits'
        if args.no_interpolate:
            cmd = cmd + ' -ninter'
        if args.orig_res:
            cmd = cmd + ' -ores'

        run_cmd(cmd)

    stats.obstats2table(args.output_dir,sub_list[:,0])


    sys.exit(0)








