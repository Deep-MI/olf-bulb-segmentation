import subprocess
import shlex
import sys
import argparse
import os
import shutil
sys.path.append('../')
sys.path.append('../../')

from ob_pipeline.utils import stats


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
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-sub_list", "--sublist", help="Subject List", required=True)

    parser.add_argument("-out_dir", "--output_dir",
                        help="Main output directory where models results are going to be store", required=True)

    parser.add_argument("-data_dir", "--data_dir", type=str, help="data directory", required=True)

    parser.add_argument('-batch', "--batch_size", type=int,
                        help='Batch size for inference by default is 16', required=False, default=16)
    parser.add_argument('-model', "--model", type=int,
                        help='model number', required=False, default=5)


    parser.add_argument('-gpu_id', "--gpu_id", type=int,
                        help='GPU device name to run model', required=False, default=0)
    parser.add_argument('-no_cuda', "--no_cuda", action='store_true',
                        help='Disable CUDA (no GPU usage, inference on CPU)', required=False)
    parser.add_argument('-logits', "--save_logits", action='store_true',
                        help='Save logits', required=False)

    parser.add_argument('-rs', '--rs', action='store_true', help='compute RS statistics', required=False)

    parser.add_argument('-hires', '--hires', action='store_true', help='Upsample crop region', required=False)


    parser.add_argument('-loc_dir', '--loc_dir', help='Localization weights directory', required=False,
                        default='./LocModels')
    parser.add_argument('-loc_arc', '--loc_arc', help='Localization architecture', required=False, default='FastSurferCNN')

    parser.add_argument('-seg_dir', '--seg_dir', help='Segmentation weights directory', required=False,
                        default='./SegModels')
    parser.add_argument('-seg_arc', '--seg_arc', help='Segmentation architecture', required=False,
                        default='AttCDFNet')

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
        cmd='python3 ./run_pipeline.py -in_img {} -out_dir {} -sid {} -batch {} -gpu_id {} ' \
            '-loc_dir {} -loc_arc {} -seg_dir {} -seg_arc {} -model {}'.format(sub_list[i,1],args.output_dir,
                                                                    sub_list[i,0],args.batch_size,
                                                                    args.gpu_id,args.loc_dir,
                                                                    args.loc_arc,args.seg_dir,
                                                                    args.seg_arc,args.model)

        if args.no_cuda:
            cmd =cmd +' -no_cuda'
        if args.save_logits:
            cmd = cmd + ' -logits'
        if args.rs:
            cmd = cmd + ' -rs'
        if args.hires:
            cmd = cmd + ' -hires'

        run_cmd(cmd)

    stats.obstats2table(args.output_dir,sub_list[:,0])

    # if os.path.isdir("./tmp"):
    #     shutil.rmtree("./tmp")

    sys.exit(0)








