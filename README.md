# Overview

This repository contains the tool designed for segmenting the olfactory bulb on high-resolutional (0.7 or 0.8mm Isotropic) whole brain T2-Weighted MRI.

![](/images/pipeline.png)

* First publicly available deep learning pipeline to segment the olfactory bulbs (OBs) in sub-millimeter T2-weighted whole-brain MRI.

* Rigorous validation in the Rhineland Study - an ongoing large population-based cohort study - in terms of segmentation accuracy, stability and reliability of volume estimates, as well as sensitivity to replicate known OB volume associations (e.g. age effects).

* Good generalizability to an unseen heterogeneous independent dataset (the Human Connectome Project).

* Robustness even for individuals without apparent OBs, as can be encountered in large cohort studies.


## Tool installation
If the tool is run for the first time run the following steps:

 1. Open Terminal
 2. Change the current working directory to the location where you want the cloned directory 
 3. Type `git-clone https://github.com/Deep-MI/olf-bulb-segmentation.git`  or download .zip file from the github repository 
 4. Download the model weights from the following [link](https://nextcloud.dzne.de/index.php/s/QaYpocJn9HFN7jp) into the same location were the repository was cloned.
 5. Installed the python libraries mention on the requirements file (see [requirements.txt](./requirements.txt))  

**The tool also has a docker version**. Docker images are really easy to run and to make. They eliminate the challenges of setting up your code to work in different environments. Your Docker image should run the same no matter where it is running.
  For more information check the [README.md](./docker/README.md) in the docker directory.

## Running the tool

The repository contains all the source code and modules needed to run the scripts. 
The main script is called run_pipeline.py within which certain options can be selected and set via the command line:


#### Required Arguments
 * `--in_img,-in`: T2 image path 
 * `--output_dir,-out`: Main output directory where pipeline results are going to be stored
 * `--sub_id,-sid`: subject_id; All generated outputs are stored under the subject_id folder as follows: */output_dir/sub_id* 

#### Optional Arguments Pipeline setup
 * `--no_interpolate, -ninter`: Flag to disable the interpolation of the input scans to the default training resolution of 0.8mm isotropic
 * `--order, -order`: Interpolation order to used if input scan is interpolated (0=nearest,1=linear(default),2=quadratic,3=cubic)
 * `--save_logits, -logits`: Flag to additionally save segmentation logits maps as h5 file
 * `--model, -model`: *AttFastSurferCNN* model to be run by default the pipeline runs all 4 *AttFastSurferCNN* models (1 = model 1, 2 = model 2, 3 = model 3, 4 = model 4, 5= all models (default))
 * `--orig_res, -ores`: Flag to upsample or downsample the OB segmentation to the native input image resolution by default the pipeline produces a segmentation with a 0.8mm isotropic resolution.
 * `--loc_dir, -loc_dir`: Localization weights directory (default = *./LocModels* , the pipeline expects the model weights to be in the same directory as the source code)
 * `--seg_dir, -seg_dir`: Segmentation weights directory  (default = *./SegModels* , the pipeline expects the model weights to be in the same directory as the source code)
 
#### Optional Arguments System Setup
 * `--batch_size,-batch`: Batch size for inference (default = 8, the batch size depends of the size of the GPU or CPU. Lower this parameter to reduce memory requirements ) 
 * `--no_cuda, -ncuda`: Flag to disable CUDA usage (no GPU usage, inference on CPU)
 * `--gpu_id, -gpu` : GPU device name to run model (default = 0) 



**Example**
```
# Run paper implementation 
python3 run_pipeline.py -in /input/t2/image -out /directory/to/save/output -sid subject

# Run the pipeline natively at a different resolution to the default one (0.8 Isotropic)
python3 run_pipeline.py -in /input/t2/image -out /directory/to/save/output -sid subject -ninter
    
# Run the pipeline at the default resolution (0.8 Isotropic) but mapped segmentation to the native input image resolution
python3 run_pipeline.py -in /input/t2/image -out /directory/to/save/output -sid subject -ores

# Run paper implementation on cpu
python3 run_pipeline.py -in /input/t2/image -out /directory/to/save/output -sid subject -ncuda

```

**Sample Case**
The [link](https://nextcloud.dzne.de/index.php/s/QaYpocJn9HFN7jp) with model weights additionally includes a sample t2 image for testing the pipeline. To test the pipeline run: 
```
python3 run_pipeline.py -in path/to/sample/T2_sample.nii.gz -out /directory/to/save/output -sid sample
```

## Reference

If you use this tool please cite:

Estrada, Santiago, et al. "Automated olfactory bulb segmentation on high resolutional T2-weighted MRI." NeuroImage (2021). [https://doi.org/10.1016/j.neuroimage.2021.118464](https://www.sciencedirect.com/science/article/pii/S1053811921007370)
```
@article{estrada2021automated,
  title={Automated Olfactory Bulb Segmentation on High Resolutional T2-Weighted MRI},
  author={Estrada, Santiago and Lu, Ran and Diers, Kersten and Zeng, Weiyi and Ehses, Philipp and St{\"o}cker, Tony and Breteler, Monique MB and Reuter, Martin},
  journal={NeuroImage},
  pages={118464},
  year={2021},
  publisher={Elsevier}
}

```

--------
For any questions and feedback, feel free to contact santiago.estrada(at).dzne.de<br/>

--------