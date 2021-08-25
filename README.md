# Overview

This repository contains the tool designed for segmenting the olfactory bulb on high-resolutional (0.7 or 0.8mm Isotropic) whole brain T2-Weighted MRI. [Paper](https://www.sciencedirect.com/science/article/pii/S1053811921007370)

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
 4. Installed the python libraries mention on the requirements file (see [requirements.txt](./requirements.txt))  

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
 * `--save_logits, -logits`: Flag to additionally save segmentation prediction logits maps as h5 file
 * `--model, -model`: *AttFastSurferCNN* model to be run by default the pipeline runs all 4 *AttFastSurferCNN* models (1 = model 1, 2 = model 2, 3 = model 3, 4 = model 4, 5= all models (default))
 * `--orig_res, -ores`: Flag to upsample or downsample the OB segmentation to the native input image resolution by default the pipeline produces a segmentation with a 0.8mm isotropic resolution.
 * `--loc_dir, -loc_dir`: Localization weights directory (default = *./LocModels* , the pipeline expects the model weights to be in the same directory as the source code)
 * `--seg_dir, -seg_dir`: Segmentation weights directory  (default = *./SegModels* , the pipeline expects the model weights to be in the same directory as the source code)
 
#### Optional Arguments System Setup
 * `--batch_size,-batch`: Batch size for inference (default = 8, the batch size depends of the size of the GPU or CPU. Lower this parameter to reduce memory requirements ) 
 * `--no_cuda, -ncuda`: Flag to disable CUDA usage (no GPU usage, inference on CPU)
 * `--gpu_id, -gpu` : GPU device name to run model (default = 0) 



**Example** Note for the commands to work on the terminal, you need to change the current working directory to the location where you cloned the repository.
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

**Quick and easy - OB segmentation**

To evaluate the pipeline, we provide a colab notebook for quick and easy use; no programming experience required. Just click the google colab icon to access the notebook <a href="https://colab.research.google.com/github/Deep-MI/olf-bulb-segmentation/blob/main/OB_pipeline_test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<br/>
You required a google account to interact with the notebook. For more information on how to use colab in neuroimaging see our group [tutorials](https://github.com/Deep-MI/FastSurfer/tree/master/Tutorial).

## Output
The pipeline generates three type of output as presented in the following scheme:

```  bash
#Output Scheme 
|-- output_dir                                   
    |-- sub_id
        |-- mri (interpolated scans and segmentation maps)
           |-- orig.nii.gz (Input image to the pipeline after interpolation and intensities conform)
           |-- orig_crop.nii.gz (Only created if --orig_res flag is used, T2 from the region of interest at the native image resolution)
           |-- ob_seg.nii.gz (OB prediction map)
           |-- loc_orig.nii.gz (Input image to the localization network)
           |-- loc_heatmap.nii.gz (Localization prediction map)
           |-- ob_seg_logits.h5 (Only created if --save_logits flag is used, segmentation prediction logits maps)          
        |-- QC (Quality control images these images are created only for a fast assessment of the segmentation for a detailed QC is still recommended to open the segmentation map)
           |-- coronal_screenshot.png 
           |-- overall_screenshot.png
        |-- stats                                                 
           |-- segmentation_stats.csv (Volume predictions and warning flags)
           |-- localization_stats.csv (Localization stats used for croping the region of interest and warning flags)         
 ``` 
 
**Image Biomarkers**

For more information on the pipeline image biomarkers reported in the CSV files please check the document [variables.pdf](/to/do)

**Quality Control Image Example**

By default, the tool creates two images for visually controlling the input scan and predicted segmentation, one shown below. (blue: Left OB, red : Right OB).
The other one shows a sagittal, coronal and axial view of the prediction map around the centroid of mass.
 
![](/images/qc_example.png)

 
## Reference

If you use this tool please cite:

Estrada, Santiago, et al. "Automated olfactory bulb segmentation on high resolutional T2-weighted MRI." NeuroImage (2021). https://doi.org/10.1016/j.neuroimage.2021.118464
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
